from pathlib import Path
import torch

import hydra
import omegaconf
import pandas as pd
import numpy as np
import librosa


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    df = pd.read_csv("./audio_diff_hifi.csv")
    df = df.nlargest(n=10, columns=["diff"])
    audio_dir = Path(cfg.path.hifi_captain.audio_dir).expanduser()
    hubert_feature_prj_output_dir = Path(
        cfg.path.hifi_captain.hubert_feature_prj_output_dir
    ).expanduser()
    hubert_encoder_output_dir = Path(
        cfg.path.hifi_captain.hubert_encoder_output_dir
    ).expanduser()

    for row in df.iterrows():
        audio_path = row[1]["audio_path"]
        audio_path_cut = row[1]["audio_path_cut"]

        wav, _ = librosa.load(str(audio_path), sr=cfg["data"]["audio"]["sr"])
        wav = wav[: wav.shape[0] - (wav.shape[0] % (cfg["data"]["audio"]["sr"] // 100))]
        wav = np.pad(
            wav,
            (0, int(cfg["data"]["audio"]["sr"] // 100 * 2)),
            mode="constant",
            constant_values=0,
        )
        wav_input = torch.from_numpy(wav).unsqueeze(0)

        wav_cut, _ = librosa.load(str(audio_path_cut), sr=cfg["data"]["audio"]["sr"])
        wav_cut = wav_cut[
            : wav_cut.shape[0]
            - (wav_cut.shape[0] % (cfg["data"]["audio"]["sr"] // 100))
        ]
        wav_cut = np.pad(
            wav_cut,
            (0, int(cfg["data"]["audio"]["sr"] // 100 * 2)),
            mode="constant",
            constant_values=0,
        )
        wav_cut_input = torch.from_numpy(wav_cut).unsqueeze(0)

        hubert_feature_prj_path = Path(
            str(audio_path_cut)
            .replace(str(audio_dir), str(hubert_feature_prj_output_dir))
            .replace(".wav", ".npy")
        )
        hubert_feature_prj = np.load(str(hubert_feature_prj_path))

        hubert_encoder_path = Path(
            str(audio_path_cut)
            .replace(str(audio_dir), str(hubert_encoder_output_dir))
            .replace(".wav", ".npy")
        )
        hubert_encoder = np.load(str(hubert_encoder_path))

        print(f"wav_input: {wav_input.shape[1] / 16000}")
        print(f"wav_cut_input: {wav_cut_input.shape[1] / 16000}")
        print(f"hubert_feature_prj: {hubert_feature_prj.shape[0] / 50}")
        print(f"hubert_encoder: {hubert_encoder.shape[0] / 50}")
        print()
        print()


if __name__ == "__main__":
    main()
