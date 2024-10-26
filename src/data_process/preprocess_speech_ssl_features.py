import pickle
from pathlib import Path

import hydra
import librosa
import numpy as np
import omegaconf
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModel


class Preprocessor:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        process_data: str,
        process_model: str,
        debug: bool,
    ) -> None:
        self.cfg = cfg
        self.process_data = process_data
        self.process_model = process_model
        self.debug = debug

        if process_model == "hubert":
            self.model = AutoModel.from_pretrained("rinna/japanese-hubert-base")
        elif process_model == "wav2vec2":
            self.model = AutoModel.from_pretrained("rinna/japanese-wav2vec2-base")
        elif process_model == "data2vec":
            self.model = AutoModel.from_pretrained("rinna/japanese-data2vec-audio-base")
        else:
            raise ValueError("model was not found.")

        self.model = self.model.cuda().eval()

        self.df = pd.read_csv(str(Path(cfg.path[process_data].df_path).expanduser()))
        self.audio_dir = Path(cfg.path[process_data].audio_dir).expanduser()
        self.conv_feature_dir = Path(
            cfg.path[process_data][process_model].conv_feature_dir
        ).expanduser()
        self.layer_feature_dir = Path(
            cfg.path[process_data][process_model].layer_feature_dir
        ).expanduser()
        self.kmeans_dir = Path(
            cfg.path[cfg.model.decoder.speech_ssl.kmeans][process_model].kmeans_dir
        ).expanduser()
        self.layer_feature_cluster_dir = (
            Path(
                cfg.path[process_data][process_model].layer_feature_cluster_dir
            ).expanduser()
            / cfg.model.decoder.speech_ssl.kmeans
        )

        if debug:
            self.conv_feature_dir = self.change_dirpath_for_debug(self.conv_feature_dir)
            self.layer_feature_dir = self.change_dirpath_for_debug(
                self.layer_feature_dir
            )
            self.kmeans_dir = self.change_dirpath_for_debug(self.kmeans_dir)
            self.layer_feature_cluster_dir = self.change_dirpath_for_debug(
                self.layer_feature_cluster_dir
            )

    def change_dirpath_for_debug(self, dirpath) -> Path:
        return Path(str(dirpath).replace("dataset", "dataset_debug"))

    def extend_path(self, dirpath: Path, row: pd.Series, suffix: str) -> Path:
        if self.process_data == "kablab":
            path = dirpath / row["speaker"] / f'{row["filename"]}{suffix}'
        elif self.process_data == "hifi_captain":
            path = (
                dirpath
                / row["speaker"]
                / "wav"
                / row["parent_dir"]
                / f'{row["filename"]}{suffix}'
            )
        elif self.process_data == "jvs":
            path = (
                dirpath
                / row["speaker"]
                / row["data"]
                / "wav24kHz16bit"
                / f'{row["filename"]}{suffix}'
            )
        return path

    def save_numerical_features(self, layer_index_lst: list[int]) -> None:
        print(f"save_numerical_features: {self.process_data}, {self.process_model}")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            audio_path = self.extend_path(self.audio_dir, row, ".wav")
            if not audio_path.exists():
                continue

            wav, _ = librosa.load(str(audio_path), sr=self.cfg.data.audio.sr)
            wav = wav[: wav.shape[0] - (wav.shape[0] % (self.cfg.data.audio.sr // 100))]
            wav = np.pad(
                wav,
                (0, int(self.cfg.data.audio.sr // 100 * 2)),
                mode="constant",
                constant_values=0,
            )
            wav_input = torch.from_numpy(wav).unsqueeze(0).cuda()
            with torch.no_grad():
                feature_extractor_output = self.model.feature_extractor(wav_input)
                feature_extractor_output = feature_extractor_output.permute(0, 2, 1)

                # マスク対象出力
                conv_feature = self.model.feature_projection(feature_extractor_output)

                # wav2vec2.0とdata2vec-audioは余分な出力も返るので0番目だけを取得
                if self.process_model in ["wav2vec2", "data2vec"]:
                    conv_feature = conv_feature[0]

                output = self.model(
                    input_values=wav_input,
                    output_hidden_states=True,
                )

            conv_feature_ndarray = conv_feature.cpu().squeeze(0).numpy()
            conv_feature_path = self.extend_path(self.conv_feature_dir, row, ".npy")
            conv_feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(conv_feature_path), conv_feature_ndarray)

            for layer_index in layer_index_lst:
                layer_feature_path = self.extend_path(
                    self.layer_feature_dir / str(layer_index), row, ".npy"
                )
                layer_feature_path.parent.mkdir(parents=True, exist_ok=True)
                layer_feature = (
                    output.hidden_states[layer_index].cpu().squeeze(0).numpy()
                )
                np.save(str(layer_feature_path), layer_feature)

            if self.debug:
                break

    def save_kmeans(self, layer_index: int, n_clusters: int) -> None:
        if self.process_data != self.cfg.model.decoder.speech_ssl.kmeans:
            return

        print(
            f"save_kmeans: {self.process_data}, {self.process_model}, {layer_index=}, {n_clusters=}"
        )

        df_kmeans = self.df.copy()

        # kablabのtrainかつatrのみを利用
        df_kmeans = df_kmeans.loc[
            (df_kmeans["data_split"] == "train") & (df_kmeans["corpus"] == "ATR")
        ].reset_index(drop=True)

        layer_feature_lst = []
        for i, row in tqdm(df_kmeans.iterrows(), total=len(df_kmeans)):
            layer_feature_path = self.extend_path(
                self.layer_feature_dir / str(layer_index), row, ".npy"
            )
            if not layer_feature_path.exists():
                continue
            layer_feature = np.load(str(layer_feature_path))
            layer_feature_lst.append(layer_feature)

        layer_feature_all = np.concatenate(layer_feature_lst, axis=0)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            batch_size=10000,
            compute_labels=False,
            random_state=42,
            verbose=1,
            max_iter=10000,
        ).fit(layer_feature_all)

        kmeans_path = self.kmeans_dir / str(layer_index) / f"{n_clusters}.pickle"
        kmeans_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(kmeans_path), "wb") as f:
            pickle.dump(kmeans, f)

    def save_clusters(self, layer_index: int, n_clusters: int):
        kmeans_path = self.kmeans_dir / str(layer_index) / f"{n_clusters}.pickle"

        if not kmeans_path.exists():
            raise FileNotFoundError(f"{str(kmeans_path)} was not Found.")

        print(
            f"save_clusters: {self.process_data}, {self.process_model}, {layer_index=}, {n_clusters=}"
        )

        with open(str(kmeans_path), "rb") as f:
            kmeans = pickle.load(f)

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            layer_feature_path = self.extend_path(
                self.layer_feature_dir / str(layer_index), row, ".npy"
            )
            layer_feature_cluster_path = self.extend_path(
                self.layer_feature_cluster_dir / str(layer_index) / str(n_clusters),
                row,
                ".npy",
            )
            if not layer_feature_path.exists():
                continue

            layer_feature = np.load(str(layer_feature_path))
            layer_feature_cluster = kmeans.predict(layer_feature)
            layer_feature_cluster += 1  # for padding index
            layer_feature_cluster_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(layer_feature_cluster_path), layer_feature_cluster)

            if self.debug:
                break


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    layer_index_lst = [8, 10, 12]
    n_clusters_lst = [100, 200, 300]
    for process_data in ["kablab", "jvs", "hifi_captain"]:
        for process_model in ["hubert"]:
            preprocessor = Preprocessor(
                cfg=cfg,
                process_data=process_data,
                process_model=process_model,
                debug=False,
            )
            preprocessor.save_numerical_features(layer_index_lst)
            for layer_index in layer_index_lst:
                for n_clusters in n_clusters_lst:
                    preprocessor.save_kmeans(layer_index, n_clusters)
                    preprocessor.save_clusters(layer_index, n_clusters)


if __name__ == "__main__":
    main()
