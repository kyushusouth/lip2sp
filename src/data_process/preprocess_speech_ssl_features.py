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

        self.model = (
            AutoModel.from_pretrained(f"rinna/japanese-{process_model}-base")
            .cuda()
            .eval()
        )
        self.df = pd.read_csv(str(Path(cfg.path[process_data].df_path).expanduser()))
        self.audio_dir = Path(cfg.path[process_data].audio_dir).expanduser()
        self.conv_feature_dir = Path(
            cfg.path[process_data][process_model].conv_feature_dir
        ).expanduser()
        self.final_feature_dir = Path(
            cfg.path[process_data][process_model].final_feature_dir
        ).expanduser()
        self.kmeans_path = (
            Path(
                cfg.path[cfg.model.decoder.speech_ssl.kmeans][process_model].kmeans_dir
            ).expanduser()
            / f"{self.cfg.model.decoder.speech_ssl.n_clusters}.pickle"
        )
        self.final_feature_cluster_dir = (
            Path(
                cfg.path[process_data][process_model].final_feature_cluster_dir
            ).expanduser()
            / cfg.model.decoder.speech_ssl.kmeans
            / str(cfg.model.decoder.speech_ssl.n_clusters)
        )

        if debug:
            self.conv_feature_dir = self.change_dirpath_for_debug(self.conv_feature_dir)
            self.final_feature_dir = self.change_dirpath_for_debug(
                self.final_feature_dir
            )
            self.kmeans_path = self.change_dirpath_for_debug(self.kmeans_path)
            self.final_feature_cluster_dir = self.change_dirpath_for_debug(
                self.final_feature_cluster_dir
            )

        print(f"{type(self.model)=}")
        print(f"{self.audio_dir=}")
        print(f"{self.conv_feature_dir=}")
        print(f"{self.final_feature_dir=}")
        print(f"{self.kmeans_path=}")
        print(f"{self.final_feature_cluster_dir=}")

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

    def save_numerical_features(self) -> None:
        print(f"save_numerical_features: {self.process_data}, {self.process_model}")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            audio_path = self.extend_path(self.audio_dir, row, ".wav")
            conv_feature_path = self.extend_path(self.conv_feature_dir, row, ".npy")
            final_feature_path = self.extend_path(self.final_feature_dir, row, ".npy")
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
                if self.process_model in ["wav2vec2", "data2vec-audio"]:
                    conv_feature = conv_feature[0]

                # 最終出力
                final_feature = self.model.encoder(conv_feature).last_hidden_state

            conv_feature_ndarray = conv_feature.cpu().squeeze(0).numpy()
            final_feature_ndarray = final_feature.cpu().squeeze(0).numpy()  # (T, C)
            conv_feature_path.parent.mkdir(parents=True, exist_ok=True)
            final_feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(conv_feature_path), conv_feature_ndarray)
            np.save(str(final_feature_path), final_feature_ndarray)

            if self.debug:
                break

    def save_kmeans(self) -> None:
        if self.process_data != self.cfg.model.decoder.speech_ssl.kmeans:
            return

        print(f"save_kmeans: {self.process_data}, {self.process_model}")

        df_kmeans = self.df.copy()

        # trainかつatrのみを利用
        df_kmeans = df_kmeans.loc[
            (df_kmeans["data_split"] == "train") & (df_kmeans["corpus"] == "ATR")
        ].reset_index(drop=True)

        final_feature_list = []
        for i, row in tqdm(df_kmeans.iterrows(), total=len(df_kmeans)):
            final_feature_path = self.extend_path(self.final_feature_dir, row, ".npy")
            if not final_feature_path.exists():
                continue
            final_feature = np.load(str(final_feature_path))
            final_feature_list.append(final_feature)

        final_feature_all = np.concatenate(final_feature_list, axis=0)
        kmeans = MiniBatchKMeans(
            n_clusters=self.cfg.model.decoder.speech_ssl.n_clusters,
            init="k-means++",
            batch_size=10000,
            compute_labels=False,
            random_state=42,
            verbose=1,
            max_iter=10000,
        ).fit(final_feature_all)

        self.kmeans_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.kmeans_path), "wb") as f:
            pickle.dump(kmeans, f)

    def save_clusters(self):
        if not self.kmeans_path.exists():
            raise FileNotFoundError(f"{str(self.kmeans_path)} was not Found.")

        print(f"save_clusters: {self.process_data}, {self.process_model}")

        with open(str(self.kmeans_path), "rb") as f:
            kmeans = pickle.load(f)

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            final_feature_path = self.extend_path(self.final_feature_dir, row, ".npy")
            final_feature_cluster_path = self.extend_path(
                self.final_feature_cluster_dir, row, ".npy"
            )
            if not final_feature_path.exists():
                continue

            final_feature = np.load(str(final_feature_path))
            final_feature_cluster = kmeans.predict(final_feature)
            final_feature_cluster += 1  # 0はパディングに使いたいので、1加算しておく
            final_feature_cluster_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(final_feature_cluster_path), final_feature_cluster)

            if self.debug:
                break


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    for process_data in ["kablab", "jvs", "hifi_captain"]:
        for process_model in ["hubert", "wav2vec2", "data2vec-audio"]:
            preprocessor = Preprocessor(
                cfg=cfg,
                process_data=process_data,
                process_model=process_model,
                debug=False,
            )
            preprocessor.save_numerical_features()
            preprocessor.save_kmeans()
            preprocessor.save_clusters()


if __name__ == "__main__":
    main()
