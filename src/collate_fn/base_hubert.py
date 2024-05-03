import random

import omegaconf
import torch

from src.data_process.utils import get_upsample, get_upsample_hubert


def adjust_seq_lengths(batch: list, cfg: omegaconf.DictConfig) -> tuple:
    (
        wav_list,
        lip_list,
        feature_list,
        feature_avhubert_list,
        feature_hubert_encoder_list,
        feature_hubert_prj_list,
        feature_hubert_cluster_list,
        spk_emb_list,
        feature_len_list,
        feature_hubert_len_list,
        lip_len_list,
        speaker_list,
        filename_list,
    ) = list(zip(*batch))

    wav_adjusted = []
    lip_adjusted = []
    feature_adjusted = []
    feature_avhubert_adjusted = []
    feature_hubert_encoder_adjusted = []
    feature_hubert_prj_adjusted = []
    feature_hubert_cluster_adjusted = []

    lip_input_len = int(cfg.training.input_sec * cfg.data.video.fps)
    upsample = get_upsample(cfg)
    feat_input_len = int(lip_input_len * upsample)
    upsample_hubert = get_upsample_hubert(cfg)
    feat_hubert_input_len = int(lip_input_len * upsample_hubert)
    wav_input_len = int(feat_input_len * cfg.data.audio.hop_length)

    for (
        wav,
        lip,
        feature,
        feature_avhubert,
        feature_hubert_encoder,
        feature_hubert_prj,
        feature_hubert_cluster,
        feature_len,
    ) in zip(
        wav_list,
        lip_list,
        feature_list,
        feature_avhubert_list,
        feature_hubert_encoder_list,
        feature_hubert_prj_list,
        feature_hubert_cluster_list,
        feature_len_list,
    ):
        if feature_len <= feat_input_len:
            w_padded = torch.zeros(wav_input_len)
            l_padded = torch.zeros(
                lip.shape[0], lip.shape[1], lip.shape[2], lip_input_len
            )
            f_padded = torch.zeros(feature.shape[0], feat_input_len)
            f_avhubert_padded = torch.zeros(feature_avhubert.shape[0], lip_input_len)
            f_hubert_encoder_padded = torch.zeros(
                feature_hubert_encoder.shape[0], feat_hubert_input_len
            )
            f_hubert_prj_padded = torch.zeros(
                feature_hubert_prj.shape[0], feat_hubert_input_len
            )
            f_hubert_cluster_padded = torch.zeros(feat_hubert_input_len)
            wav = wav[:wav_input_len]
            w_padded[: wav.shape[0]] = wav
            l_padded[..., : lip.shape[-1]] = lip
            f_padded[:, : feature.shape[-1]] = feature
            f_avhubert_padded[:, : feature_avhubert.shape[-1]] = feature_avhubert
            f_hubert_encoder_padded[:, : feature_hubert_encoder.shape[-1]] = (
                feature_hubert_encoder
            )
            f_hubert_prj_padded[:, : feature_hubert_prj.shape[-1]] = feature_hubert_prj
            f_hubert_cluster_padded[: feature_hubert_cluster.shape[0]] = (
                feature_hubert_cluster
            )
            wav = w_padded
            lip = l_padded
            feature = f_padded
            feature_avhubert = f_avhubert_padded
            feature_hubert_encoder = f_hubert_encoder_padded
            feature_hubert_prj = f_hubert_prj_padded
            feature_hubert_cluster = f_hubert_cluster_padded
        else:
            lip_start_frame = random.randint(0, lip.shape[-1] - lip_input_len - 1)
            feature_start_frame = int(lip_start_frame * upsample)
            feature_hubert_start_frame = int(lip_start_frame * upsample_hubert)
            wav_start_sample = int(feature_start_frame * cfg.data.audio.hop_length)
            wav = wav[wav_start_sample : wav_start_sample + wav_input_len]
            lip = lip[..., lip_start_frame : lip_start_frame + lip_input_len]
            feature = feature[
                :, feature_start_frame : feature_start_frame + feat_input_len
            ]
            feature_avhubert = feature_avhubert[
                :, lip_start_frame : lip_start_frame + lip_input_len
            ]
            feature_hubert_encoder = feature_hubert_encoder[
                :,
                feature_hubert_start_frame : feature_hubert_start_frame
                + feat_hubert_input_len,
            ]
            feature_hubert_prj = feature_hubert_prj[
                :,
                feature_hubert_start_frame : feature_hubert_start_frame
                + feat_hubert_input_len,
            ]
            feature_hubert_cluster = feature_hubert_cluster[
                feature_hubert_start_frame : feature_hubert_start_frame
                + feat_hubert_input_len,
            ]

        assert wav.shape[0] == wav_input_len
        assert lip.shape[-1] == lip_input_len
        assert feature.shape[-1] == feat_input_len
        assert feature_avhubert.shape[-1] == lip_input_len
        assert feature_hubert_encoder.shape[-1] == feat_hubert_input_len
        assert feature_hubert_prj.shape[-1] == feat_hubert_input_len
        assert feature_hubert_cluster.shape[-1] == feat_hubert_input_len

        wav_adjusted.append(wav)
        lip_adjusted.append(lip)
        feature_adjusted.append(feature)
        feature_avhubert_adjusted.append(feature_avhubert)
        feature_hubert_encoder_adjusted.append(feature_hubert_encoder)
        feature_hubert_prj_adjusted.append(feature_hubert_prj)
        feature_hubert_cluster_adjusted.append(feature_hubert_cluster)

    wav = torch.stack(wav_adjusted).to(torch.float32)
    lip = torch.stack(lip_adjusted).to(torch.float32)
    feature = torch.stack(feature_adjusted).to(torch.float32)
    feature_avhubert = torch.stack(feature_avhubert_adjusted).to(torch.float32)
    feature_hubert_encoder = torch.stack(feature_hubert_encoder_adjusted).to(
        torch.float32
    )
    feature_hubert_prj = torch.stack(feature_hubert_prj_adjusted).to(torch.float32)
    feature_hubert_cluster = torch.stack(feature_hubert_cluster_adjusted).to(torch.long)
    spk_emb = torch.stack(spk_emb_list).to(torch.float32)
    feature_len = torch.stack(feature_len_list).to(torch.int)
    feature_hubert_len = torch.stack(feature_hubert_len_list).to(torch.int)
    lip_len = torch.stack(lip_len_list).to(torch.int)

    return (
        wav,
        lip,
        feature,
        feature_avhubert,
        feature_hubert_encoder,
        feature_hubert_prj,
        feature_hubert_cluster,
        spk_emb,
        feature_len,
        feature_hubert_len,
        lip_len,
        speaker_list,
        filename_list,
    )
