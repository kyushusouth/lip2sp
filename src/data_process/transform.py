import pathlib

import librosa
import numpy as np
import omegaconf
import torch
import torchvision

from src.data_process.feature import wav2mel, wav2mel_avhubert


def get_upsample(cfg: omegaconf.DictConfig) -> int:
    """
    動画のfpsと音響特徴量のフレームあたりの秒数から対応関係を求める
    """
    n_mel_frames_per_sec = (
        cfg["data"]["audio"]["sr"] // cfg["data"]["audio"]["hop_length"]
    )
    upsample = n_mel_frames_per_sec // cfg["data"]["video"]["fps"]
    return upsample


def load_data(
    audio_path: pathlib.Path, video_path: pathlib.Path, cfg: omegaconf.DictConfig
) -> tuple:
    wav, _ = librosa.load(str(audio_path), sr=cfg["data"]["audio"]["sr"])
    wav = wav / np.max(np.abs(wav))  # (T,)
    feature = wav2mel(wav, cfg, ref_max=False)  # (C, T)
    feature_avhubert = wav2mel_avhubert(wav, cfg)  # (C, T)
    upsample = get_upsample(cfg)

    if video_path is not None:
        lip, _, _ = torchvision.io.read_video(
            str(video_path), pts_unit="sec", output_format="TCHW"
        )  # (T, C, H, W)
        lip = torchvision.transforms.functional.rgb_to_grayscale(lip)
    else:
        lip = torch.rand(int(feature.shape[1] * upsample), 1, 96, 96)
    lip = lip.numpy()

    data_len = min(
        int(feature.shape[1] // upsample * upsample),
        int(feature_avhubert.shape[1] // upsample * upsample),
        int(lip.shape[0] * upsample),
    )

    wav = wav[: int(cfg["data"]["audio"]["hop_length"] * data_len)]
    wav_padded = np.zeros((int(cfg["data"]["audio"]["hop_length"] * data_len)))
    wav_padded[: wav.shape[0]] = wav
    wav = wav_padded

    feature = feature[:, :data_len]
    feature_padded = np.zeros((feature.shape[0], data_len))
    feature_padded[:, : feature.shape[1]] = feature
    feature = feature_padded

    feature_avhubert = feature_avhubert[:, :data_len]
    feature_avhubert_padded = np.zeros((feature_avhubert.shape[0], data_len))
    feature_avhubert_padded[:, : feature_avhubert.shape[1]] = feature_avhubert
    feature_avhubert = feature_avhubert_padded

    lip = lip[: data_len // upsample]
    lip_padded = np.zeros((data_len // upsample, 1, 96, 96))
    lip_padded[: lip.shape[0]] = lip
    lip = lip_padded
    return wav, feature, feature_avhubert, lip
