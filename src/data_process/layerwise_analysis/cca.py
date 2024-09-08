import logging
import pickle
from pathlib import Path

import hydra
import joblib
import librosa
import numpy as np
import omegaconf
import polars as pl
import textgrids
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModel

from src.data_process.layerwise_analysis.cca_core import CCA
from src.data_process.layerwise_analysis.utils import get_epsilon_lst, get_segment_idx
from src.data_process.utils import wav2mel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_cca(
    view1_trainval: np.array,
    view1_test: np.array,
    view2_trainval: np.array,
    view2_test: np.array,
    layer_id: int,
    trainval_results: list[list],
    test_results: list[list],
    kfold: KFold,
):
    train_best_score_lst = []
    val_best_score_lst = []
    val_best_params_lst = []

    for fold, (train_index, val_index) in tqdm(
        enumerate(kfold.split(view1_trainval)),
        total=5,
        desc=f"{layer_id=}",
        position=layer_id,
        leave=False,
    ):
        view1_train = view1_trainval[train_index]
        view1_val = view1_trainval[val_index]
        view2_train = view2_trainval[train_index]
        view2_val = view2_trainval[val_index]

        cca_obj_train = CCA(
            view1_train.transpose(1, 0),
            view2_train[:, layer_id].transpose(1, 0),
        )
        score_train_lst = []
        score_val_lst = []
        params_lst = []
        for epsilon_x, epsilon_y in get_epsilon_lst():
            score_train, params = cca_obj_train.get_cca_score(
                train=True,
                epsilon_x=epsilon_x,
                epsilon_y=epsilon_y,
                mean_score=False,
            )
            proj_mat_x, proj_mat_y, x_idxs, y_idxs = params
            cca_obj_val = CCA(
                view1_val.transpose(1, 0),
                view2_val[:, layer_id].transpose(1, 0),
            )
            score_val, _ = cca_obj_val.get_cca_score(
                train=False,
                proj_mat_x=proj_mat_x,
                proj_mat_y=proj_mat_y,
                x_idxs=x_idxs,
                y_idxs=y_idxs,
                mean_score=False,
            )
            score_train_lst.append(score_train)
            score_val_lst.append(score_val)
            params_lst.append([proj_mat_x, proj_mat_y, x_idxs, y_idxs])
            trainval_results.append(
                [layer_id, fold, epsilon_x, epsilon_y, score_train, score_val]
            )

        best_index = np.argmax(score_val_lst)
        train_best_score = score_train_lst[best_index]
        val_best_score = score_val_lst[best_index]
        best_params = params_lst[best_index]
        train_best_score_lst.append(train_best_score)
        val_best_score_lst.append(val_best_score)
        val_best_params_lst.append(best_params)

    proj_mat_x, proj_mat_y, x_idxs, y_idxs = val_best_params_lst[0]
    cca_obj_test = CCA(
        view1_test.transpose(1, 0),
        view2_test[:, layer_id].transpose(1, 0),
    )
    score_test, _ = cca_obj_test.get_cca_score(
        train=False,
        proj_mat_x=proj_mat_x,
        proj_mat_y=proj_mat_y,
        x_idxs=x_idxs,
        y_idxs=y_idxs,
        mean_score=False,
    )
    test_results.append([layer_id, score_test])

    return trainval_results, test_results


class CCACalculator:
    def __init__(self, cfg: omegaconf.DictConfig, debug: bool) -> None:
        self.cfg = cfg
        self.debug = debug
        self.debug_num_samples = 20

        self.model = AutoModel.from_pretrained(
            cfg.model.decoder.hubert.model_name
        ).cuda()

        self.textgrid_dir = (
            Path(cfg.path.kablab.forced_alignment_results_dir).expanduser() / "results"
        )
        self.audio_dir = Path(cfg.path.kablab.audio_dir).expanduser()
        self.save_dir = Path(cfg.path.kablab.cca_result_dir).expanduser()

        if debug:
            self.save_dir = self.save_dir / "debug"

        self.df = pl.read_csv(str(Path(cfg.path.kablab.df_path).expanduser()))
        self.df = self.df.with_columns(
            pl.col("filename")
            .str.split("_")
            .list.get(-1)
            .str.slice(0, 1)
            .alias("set_id")
        )

    def calc_cca(
        self,
        view1_trainval: np.array,
        view1_test: np.array,
        view2_trainval: np.array,
        view2_test: np.array,
        n_layers: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        trainval_results = []
        test_results = []

        results = joblib.Parallel(n_jobs=-2)(
            joblib.delayed(calc_cca)(
                view1_trainval,
                view1_test,
                view2_trainval,
                view2_test,
                layer_id,
                trainval_results,
                test_results,
                kfold,
            )
            for layer_id in range(n_layers)
        )

        for trainval_results_layer, test_results_layer in results:
            trainval_results += trainval_results_layer
            test_results += test_results_layer

        df_trainval_results = pl.DataFrame(
            data=trainval_results,
            schema=[
                "layer_id",
                "fold",
                "epsilon_x",
                "epsilon_y",
                "score_train",
                "score_val",
            ],
            orient="row",
        )
        df_test_results = pl.DataFrame(
            data=test_results,
            schema=["layer_id", "score_test"],
            orient="row",
        )
        return df_trainval_results, df_test_results

    def get_views_mel(self, set_id_lst: list[str]) -> tuple[np.ndarray, np.ndarray]:
        mel_all = []
        hidden_states_all = []
        df = self.df.filter(pl.col("set_id").is_in(set_id_lst))

        if self.debug:
            df = df.head(self.debug_num_samples)

        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            audio_path = self.audio_dir / row["speaker"] / f"{row['filename']}.wav"

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

            mel = wav2mel(wav, self.cfg, ref_max=False)
            mel = torch.from_numpy(mel).unsqueeze(1)  # (C, 1, T)
            mel_downsampled = (
                torch.nn.functional.conv1d(
                    mel, weight=torch.ones(1, 1, 1).to(torch.float), stride=2
                )
                .squeeze(1)
                .permute(1, 0)
            )  # (T, C)

            wav_input = torch.from_numpy(wav).unsqueeze(0).cuda()
            with torch.no_grad():
                feature_dct = self.model(
                    wav_input,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = list(feature_dct.hidden_states[1:])
            for i in range(len(hidden_states)):
                hidden_states[i] = hidden_states[i].squeeze(0).cpu().numpy()  # (T, C)

            min_len = min(hidden_states[0].shape[0], mel_downsampled.shape[0])
            mel_downsampled = mel_downsampled[:min_len]
            for i in range(len(hidden_states)):
                hidden_states[i] = hidden_states[i][:min_len]

            mel_all.append(mel_downsampled)
            hidden_states_all.append(np.stack(hidden_states, axis=1))

        mel_all = np.concatenate(mel_all, axis=0)  # (T, C)
        hidden_states_all = np.concatenate(hidden_states_all, axis=0)  # (T, N, C)
        assert mel_all.shape[0] == hidden_states_all.shape[0]

        return mel_all, hidden_states_all

    def get_views_text(
        self, set_id_lst: list[str], text_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        onehot_encoder_path = (
            Path(self.cfg.path.kablab.onehot_encoder_dir).expanduser()
            / f"{text_type}.pkl"
        )
        with open(str(onehot_encoder_path), "rb") as f:
            onehot_encoder = pickle.load(f)

        text_all = []
        hidden_states_aligned_all = []
        df = self.df.filter(pl.col("set_id").is_in(set_id_lst))

        if self.debug:
            df = df.head(self.debug_num_samples)

        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            textgrid_path = (
                self.textgrid_dir / row["speaker"] / f"{row['filename']}.TextGrid"
            )
            audio_path = self.audio_dir / row["speaker"] / f"{row['filename']}.wav"

            if not textgrid_path.exists() or not audio_path.exists():
                continue

            textgrid = textgrids.TextGrid(str(textgrid_path))
            alignment = []
            for item in textgrid[text_type]:
                token = item.text
                if token:
                    start_time = float(item.xmin)
                    end_time = float(item.xmax)
                    alignment.append([token, start_time, end_time])

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
                feature_dct = self.model(
                    wav_input,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = list(feature_dct.hidden_states[1:])
            for i in range(len(hidden_states)):
                hidden_states[i] = hidden_states[i].squeeze(0).cpu().numpy()  # (T, C)

            hidden_states_aligned = []
            for hidden_state in hidden_states:
                len_utt = len(alignment)
                hidden_state_pooled = []

                for token, start_time, end_time in alignment:
                    segment_idx = get_segment_idx(
                        start_time=start_time,
                        end_time=end_time,
                        len_utt=len_utt,
                        stride_sec=1 / 50,
                        offset=text_type == "phones",
                    )
                    hidden_state_pooled.append(hidden_state[segment_idx].mean(axis=0))

                hidden_states_aligned.append(np.stack(hidden_state_pooled, axis=0))

                assert (
                    len(alignment) == hidden_states_aligned[-1].shape[0]
                ), "sequence length must be same."

            text = np.array([text for text, _, _ in alignment], dtype=str).reshape(
                -1, 1
            )
            text = onehot_encoder.transform(text).toarray()  # (T, C)
            text_all.append(text)

            hidden_state_aligned = np.stack(hidden_states_aligned, axis=1)  # (T, N, C)
            hidden_states_aligned_all.append(hidden_state_aligned)

        text_all = np.concatenate(text_all, axis=0)  # (T, C)
        hidden_states_aligned_all = np.concatenate(
            hidden_states_aligned_all, axis=0
        )  # (T, N, C)
        assert text_all.shape[0] == hidden_states_aligned_all.shape[0]

        return text_all, hidden_states_aligned_all

    def cca_mel(self, trainval_set_id_lst: list[str], test_set_id_lst: list[str]):
        mel_trainval, hidden_states_trainval = self.get_views_mel(trainval_set_id_lst)
        mel_test, hidden_states_test = self.get_views_mel(test_set_id_lst)

        df_trainval_results, df_test_results = self.calc_cca(
            view1_trainval=mel_trainval,
            view1_test=mel_test,
            view2_trainval=hidden_states_trainval,
            view2_test=hidden_states_test,
            n_layers=hidden_states_trainval.shape[1],
        )
        save_dir = self.save_dir / "mel"
        save_dir.mkdir(parents=True, exist_ok=True)
        df_trainval_results.write_csv(str(save_dir / "trainval_results.csv"))
        df_test_results.write_csv(str(save_dir / "test_results.csv"))

    def cca_text(
        self, text_type: str, trainval_set_id_lst: list[str], test_set_id_lst: list[str]
    ):
        text_trainval, hidden_states_trainval = self.get_views_text(
            trainval_set_id_lst, text_type
        )
        text_test, hidden_states_test = self.get_views_text(test_set_id_lst, text_type)
        df_trainval_results, df_test_results = self.calc_cca(
            view1_trainval=text_trainval,
            view1_test=text_test,
            view2_trainval=hidden_states_trainval,
            view2_test=hidden_states_test,
            n_layers=hidden_states_trainval.shape[1],
        )
        save_dir = self.save_dir / text_type
        save_dir.mkdir(parents=True, exist_ok=True)
        df_trainval_results.write_csv(str(save_dir / "trainval_results.csv"))
        df_test_results.write_csv(str(save_dir / "test_results.csv"))


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    cca = CCACalculator(cfg, debug=True)
    cca.cca_mel(["a", "b", "c"], ["d"])
    cca.cca_text("phones", ["a", "b", "c"], ["d"])


if __name__ == "__main__":
    main()
