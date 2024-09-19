import random

import omegaconf
import torch

from src.data_process.utils import get_upsample, get_upsample_speech_ssl


def padding(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if len(x.shape) == 1:
        x_padded = torch.zeros(max_len)
        x_padded[: x.shape[0]] = x
    elif len(x.shape) == 2:
        x_padded = torch.zeros(x.shape[0], max_len)
        x_padded[:, : x.shape[1]] = x
    elif len(x.shape) == 4:
        x_padded = torch.zeros(x.shape[0], x.shape[1], x.shape[2], max_len)
        x_padded[..., : x.shape[-1]] = x
    return x_padded


def trimming(x: torch.Tensor, start: int, duration: int) -> torch.Tensor:
    if len(x.shape) == 1:
        x_trimmed = x[start : start + duration]
    elif len(x.shape) == 2:
        x_trimmed = x[:, start : start + duration]
    elif len(x.shape) == 4:
        x_trimmed = x[..., start : start + duration]
    return x_trimmed


def adjust_seq_lengths(batch: list, cfg: omegaconf.DictConfig) -> tuple:
    (
        wav,
        lip,
        feature,
        hubert_layer_feature_cluster,
        spk_emb,
        feature_len,
        feature_ssl_len,
        lip_len,
        speaker,
        filename,
    ) = list(zip(*batch))

    lip_input_len = int(cfg.training.input_sec * cfg.data.video.fps)
    upsample = get_upsample(cfg)
    feat_input_len = int(lip_input_len * upsample)
    upsampel_speech_ssl = get_upsample_speech_ssl(cfg)
    feat_speech_ssl_len = int(lip_input_len * upsampel_speech_ssl)
    wav_input_len = int(feat_input_len * cfg.data.audio.hop_length)

    wav_adjusted = []
    lip_adjusted = []
    feature_adjusted = []
    hubert_layer_feature_cluster_adjusted = []

    for (
        w,
        l,
        f,
        hubert_layer_f_cluster,
        f_len,
    ) in zip(
        wav,
        lip,
        feature,
        hubert_layer_feature_cluster,
        feature_len,
    ):
        if f_len <= feat_input_len:
            w = padding(w, wav_input_len)
            l = padding(l, lip_input_len)
            f = padding(f, feat_input_len)
            hubert_layer_f_cluster = padding(
                hubert_layer_f_cluster, feat_speech_ssl_len
            )
        else:
            lip_start_frame = random.randint(0, l.shape[-1] - lip_input_len - 1)
            feature_start_frame = int(lip_start_frame * upsample)
            feature_speech_ssl_start_frame = int(lip_start_frame * upsampel_speech_ssl)
            wav_start_sample = int(feature_start_frame * cfg.data.audio.hop_length)

            w = trimming(w, wav_start_sample, wav_input_len)
            l = trimming(l, lip_start_frame, lip_input_len)
            f = trimming(f, feature_start_frame, feat_input_len)
            hubert_layer_f_cluster = trimming(
                hubert_layer_f_cluster,
                feature_speech_ssl_start_frame,
                feat_speech_ssl_len,
            )

        wav_adjusted.append(w)
        lip_adjusted.append(l)
        feature_adjusted.append(f)
        hubert_layer_feature_cluster_adjusted.append(hubert_layer_f_cluster)

    wav = torch.stack(wav_adjusted).to(torch.float32)
    lip = torch.stack(lip_adjusted).to(torch.float32)
    feature = torch.stack(feature_adjusted).to(torch.float32)
    hubert_layer_feature_cluster = torch.stack(
        hubert_layer_feature_cluster_adjusted
    ).to(torch.long)
    spk_emb = torch.stack(spk_emb).to(torch.float32)
    feature_len = torch.stack(feature_len).to(torch.int)
    feature_ssl_len = torch.stack(feature_ssl_len).to(torch.int)
    lip_len = torch.stack(lip_len).to(torch.int)

    return (
        wav,
        lip,
        feature,
        hubert_layer_feature_cluster,
        spk_emb,
        feature_len,
        feature_ssl_len,
        lip_len,
        speaker,
        filename,
    )
