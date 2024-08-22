import pickle
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict
from pathlib import Path

import hydra
import librosa
import numpy as np
import omegaconf
import textgrids
import torch
from tqdm import tqdm
from transformers import AutoModel

from src.data_process.layerwise_analysis.utils import get_segment_idx, get_epsilon_lst
from src.data_process.layerwise_analysis.cca_core import CCA


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    hubert = AutoModel.from_pretrained(cfg.model.decoder.hubert.model_name).cuda()

    textgrid_dir = (
        Path(cfg.path.kablab.forced_alignment_results_dir).expanduser() / "results"
    )
    audio_dir = Path(cfg.path.kablab.audio_dir).expanduser()

    text_types = ["phones", "words"]

    onehot_encoder_dct = {}
    for text_type in text_types:
        onehot_encoder_path = (
            Path(cfg.path.kablab.onehot_encoder_dir).expanduser() / f"{text_type}.pkl"
        )
        if not onehot_encoder_path.exists():
            continue

        with open(
            str(onehot_encoder_path),
            "rb",
        ) as f:
            onehot_encoder_dct[text_type] = pickle.load(f)

    textgrid_data_path_lst = list(textgrid_dir.glob("**/*.TextGrid"))
    text_all = defaultdict(list)
    hidden_states_all = defaultdict(list)
    for textgrid_data_path in tqdm(textgrid_data_path_lst):
        textgrid = textgrids.TextGrid(str(textgrid_data_path))

        audio_path = (
            audio_dir
            / textgrid_data_path.parents[0].name
            / f"{textgrid_data_path.stem}.wav"
        )
        if not audio_path.exists():
            raise FileNotFoundError()
        wav, _ = librosa.load(str(audio_path), sr=cfg.data.audio.sr)

        alignment_dct = defaultdict(list)
        for text_type in text_types:
            if text_type not in onehot_encoder_dct:
                continue
            for item in textgrid[text_type]:
                token = item.text
                if token:
                    start_time = float(item.xmin)
                    end_time = float(item.xmax)
                    alignment_dct[text_type].append([token, start_time, end_time])

        wav, _ = librosa.load(str(audio_path), sr=cfg.data.audio.sr)
        wav = wav[: wav.shape[0] - (wav.shape[0] % (cfg.data.audio.sr // 100))]
        wav = np.pad(
            wav,
            (0, int(cfg.data.audio.sr // 100 * 2)),
            mode="constant",
            constant_values=0,
        )
        wav_input = torch.from_numpy(wav).unsqueeze(0).cuda()
        with torch.no_grad():
            feature_dct = hubert(
                wav_input,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = feature_dct.hidden_states[1:]

        hidden_states_aligned: dict[str, list[torch.Tensor]] = {}
        for text_type in text_types:
            if text_type not in onehot_encoder_dct:
                continue
            hidden_states_aligned[text_type] = []
            for hidden_state in hidden_states:
                len_utt = len(alignment_dct[text_type])
                hidden_state_pooled = []
                for token, start_time, end_time in alignment_dct[text_type]:
                    segment_idx = get_segment_idx(
                        start_time=start_time,
                        end_time=end_time,
                        len_utt=len_utt,
                        stride_sec=1 / 50,
                        offset=text_type == "phones",
                    )
                    hidden_state_pooled.append(
                        hidden_state[:, segment_idx, :]
                        .mean(dim=1)
                        .cpu()
                        .squeeze(0)
                        .numpy()
                    )
                hidden_states_aligned[text_type].append(
                    np.stack(hidden_state_pooled, axis=0)
                )

        for text_type in text_types:
            if text_type not in onehot_encoder_dct:
                continue

            for hidden_state in hidden_states_aligned[text_type]:
                assert (
                    len(alignment_dct[text_type]) == hidden_state.shape[0]
                ), "sequence length must be same."

            text = np.array(
                [text for text, _, _ in alignment_dct[text_type]], dtype=str
            ).reshape(-1, 1)
            text = onehot_encoder_dct[text_type].transform(text).toarray()  # (T, C)
            text_all[text_type].append(text)

            hidden_states = np.stack(
                hidden_states_aligned[text_type], axis=1
            )  # (T, n_layers, C)
            hidden_states_all[text_type].append(hidden_states)

    for text_type in text_all.keys():
        text_all[text_type] = np.concatenate(text_all[text_type], axis=0)  # (T, C)
        hidden_states_all[text_type] = np.concatenate(
            hidden_states_all[text_type], axis=0
        )  # (T, n_layers, C)
        assert text_all[text_type].shape[0] == hidden_states_all[text_type].shape[0]

    # cca
    # subset一回分として実装する。
    # とりあえず試しにシンプルな交差検証だけ実装する。
    num_trial = 3
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    text_type = "phones"
    text_all_trainval, text_all_test, hidden_states_trainval, hidden_states_test = (
        train_test_split(
            text_all[text_type], hidden_states_all[text_type], test_size=0.2
        )
    )

    for layer_index in range(12):
        print(f"{layer_index=}")

        train_best_score_lst = []
        val_best_score_lst = []
        val_best_params_lst = []

        for fold, (train_index, val_index) in enumerate(
            kfold.split(text_all_trainval, hidden_states_trainval)
        ):
            print(f"{fold=}")

            text_all_train = text_all_trainval[train_index]
            text_all_val = text_all_trainval[val_index]
            hidden_states_train = hidden_states_trainval[train_index]
            hidden_states_val = hidden_states_trainval[val_index]

            cca_obj_train = CCA(
                text_all_train.transpose(1, 0),
                hidden_states_train[:, layer_index].transpose(1, 0),
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
                    text_all_val.transpose(1, 0),
                    hidden_states_val[:, layer_index].transpose(1, 0),
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

            best_index = np.argmax(score_val_lst)
            train_best_score = score_train_lst[best_index]
            val_best_score = score_val_lst[best_index]
            best_params = params_lst[best_index]
            train_best_score_lst.append(train_best_score)
            val_best_score_lst.append(val_best_score)
            val_best_params_lst.append(best_params)

        proj_mat_x, proj_mat_y, x_idxs, y_idxs = val_best_params_lst[0]
        cca_obj_test = CCA(
            text_all_test.transpose(1, 0),
            hidden_states_test[:, layer_index].transpose(1, 0),
        )
        score_test, _ = cca_obj_test.get_cca_score(
            train=False,
            proj_mat_x=proj_mat_x,
            proj_mat_y=proj_mat_y,
            x_idxs=x_idxs,
            y_idxs=y_idxs,
            mean_score=False,
        )

        print(score_test)
        print()


if __name__ == "__main__":
    main()
