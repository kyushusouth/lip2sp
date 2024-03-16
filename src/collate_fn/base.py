import random

import omegaconf
import torch

from src.data_process.utils import get_upsample


def adjust_seq_lengths(batch: list, cfg: omegaconf.DictConfig) -> tuple:
    (
        wav_list,
        lip_list,
        feature_list,
        feature_avhubert_list,
        spk_emb_list,
        feature_len_list,
        lip_len_list,
        speaker_list,
        filename_list,
    ) = list(zip(*batch))

    wav_adjusted = []
    lip_adjusted = []
    feature_adjusted = []
    feature_avhubert_adjusted = []

    lip_input_len = int(cfg["training"]["input_sec"] * cfg["data"]["video"]["fps"])
    upsample_scale = get_upsample(cfg)
    feat_input_len = int(lip_input_len * upsample_scale)
    wav_input_len = int(feat_input_len * cfg["data"]["audio"]["hop_length"])

    for wav, lip, feature, feature_avhubert, feature_len in zip(
        wav_list, lip_list, feature_list, feature_avhubert_list, feature_len_list
    ):
        if feature_len <= feat_input_len:
            w_padded = torch.zeros(wav_input_len)
            l_padded = torch.zeros(
                lip.shape[0], lip.shape[1], lip.shape[2], lip_input_len
            )
            f_padded = torch.zeros(feature.shape[0], feat_input_len)
            f_avhubert_padded = torch.zeros(feature_avhubert.shape[0], lip_input_len)
            wav = wav[:wav_input_len]
            w_padded[: wav.shape[0]] = wav
            l_padded[..., : lip.shape[-1]] = lip
            f_padded[:, : feature.shape[-1]] = feature
            f_avhubert_padded[:, : feature_avhubert.shape[-1]] = feature_avhubert
            wav = w_padded
            lip = l_padded
            feature = f_padded
            feature_avhubert = f_avhubert_padded
        else:
            lip_start_frame = random.randint(0, lip.shape[-1] - lip_input_len - 1)
            feature_start_frame = int(lip_start_frame * upsample_scale)
            wav_start_sample = int(
                feature_start_frame * cfg["data"]["audio"]["hop_length"]
            )
            wav = wav[wav_start_sample : wav_start_sample + wav_input_len]
            lip = lip[..., lip_start_frame : lip_start_frame + lip_input_len]
            feature = feature[
                :, feature_start_frame : feature_start_frame + feat_input_len
            ]
            feature_avhubert = feature_avhubert[
                :, lip_start_frame : lip_start_frame + lip_input_len
            ]

        assert wav.shape[0] == wav_input_len
        assert lip.shape[-1] == lip_input_len
        assert feature.shape[-1] == feat_input_len
        assert feature_avhubert.shape[-1] == lip_input_len

        wav_adjusted.append(wav)
        lip_adjusted.append(lip)
        feature_adjusted.append(feature)
        feature_avhubert_adjusted.append(feature_avhubert)

    wav = torch.stack(wav_adjusted)
    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    feature_avhubert = torch.stack(feature_avhubert_adjusted)
    spk_emb = torch.stack(spk_emb_list)
    feature_len = torch.stack(feature_len_list)
    lip_len = torch.stack(lip_len_list)
    return (
        wav,
        lip,
        feature,
        feature_avhubert,
        spk_emb,
        feature_len,
        lip_len,
        speaker_list,
        filename_list,
    )
