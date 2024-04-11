import pathlib
from pathlib import Path

import librosa
import numpy as np
import omegaconf
import torch
import torchvision
from torch.utils.data import Dataset

from src.data_process.utils import (
    get_upsample,
    get_upsample_hubert,
    wav2mel,
    wav2mel_avhubert,
)
from src.dataset.utils import (
    get_spk_emb,
    get_spk_emb_hifi_captain,
    get_spk_emb_jsut,
    get_spk_emb_jvs,
)
from src.transform.base import BaseTransform


class BaseHuBERTDataset(Dataset):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        data_path_list: list[pathlib.Path],
        transform: BaseTransform,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_path_list = data_path_list
        self.transform = transform
        self.embs = get_spk_emb(cfg)
        self.embs.update(get_spk_emb_jvs(cfg))
        self.embs.update(get_spk_emb_hifi_captain(cfg))
        self.embs.update(get_spk_emb_jsut(cfg))
        self.lip_mean = torch.from_numpy(np.array([cfg["data"]["video"]["lip_mean"]]))
        self.lip_std = torch.from_numpy(np.array([cfg["data"]["video"]["lip_std"]]))
        feat_mean_var_std = np.load(
            str(Path(cfg["path"]["vctk"]["stat_path"]).expanduser())
        )
        self.feat_mean = torch.from_numpy(feat_mean_var_std["feat_mean"])
        self.feat_std = torch.from_numpy(feat_mean_var_std["feat_std"])

    def __len__(self) -> int:
        return len(self.data_path_list)

    def __getitem__(self, index: int) -> tuple:
        audio_path = self.data_path_list[index]["audio_path"]
        video_path = self.data_path_list[index]["video_path"]
        hubert_encoder_output_path = self.data_path_list[index][
            "hubert_encoder_output_path"
        ]
        hubert_feature_prj_output_path = self.data_path_list[index][
            "hubert_feature_prj_output_path"
        ]
        hubert_cluster_path = self.data_path_list[index]["hubert_cluster_path"]
        speaker = self.data_path_list[index]["speaker"]
        filename = self.data_path_list[index]["filename"]
        spk_emb = torch.from_numpy(self.embs[speaker])

        wav, _ = librosa.load(str(audio_path), sr=self.cfg["data"]["audio"]["sr"])
        wav = wav / np.max(np.abs(wav))  # (T,)
        feature = wav2mel(wav, self.cfg, ref_max=False)  # (C, T)
        feature_avhubert = wav2mel_avhubert(wav, self.cfg)  # (C, T)
        feature_hubert_encoder = np.load(str(hubert_encoder_output_path)).T  # (C, T)
        feature_hubert_prj = np.load(str(hubert_feature_prj_output_path)).T  # (C, T)
        feature_hubert_cluster = np.load(str(hubert_cluster_path))  # (T,)
        upsample = get_upsample(self.cfg)
        upsample_hubert = get_upsample_hubert(self.cfg)

        if video_path is not None:
            lip, _, _ = torchvision.io.read_video(
                str(video_path), pts_unit="sec", output_format="TCHW"
            )  # (T, C, H, W)
            lip = torchvision.transforms.functional.rgb_to_grayscale(lip)
        else:
            lip = torch.rand(int(feature.shape[1] * upsample), 1, 96, 96)
        lip = lip.numpy()

        lip_len = min(
            int(feature.shape[1] // upsample),
            int(feature_avhubert.shape[1] // upsample),
            int(feature_hubert_encoder.shape[1] // upsample_hubert),
            int(lip.shape[0]),
        )

        wav = wav[: int(self.cfg["data"]["audio"]["hop_length"] * upsample * lip_len)]
        wav_padded = np.zeros(
            (int(self.cfg["data"]["audio"]["hop_length"] * upsample * lip_len))
        )
        wav_padded[: wav.shape[0]] = wav
        wav = wav_padded

        feature = feature[:, : int(lip_len * upsample)]
        feature_padded = np.zeros((feature.shape[0], int(lip_len * upsample)))
        feature_padded[:, : feature.shape[1]] = feature
        feature = feature_padded

        feature_avhubert = feature_avhubert[:, : int(lip_len * upsample)]
        feature_avhubert_padded = np.zeros(
            (feature_avhubert.shape[0], int(lip_len * upsample))
        )
        feature_avhubert_padded[:, : feature_avhubert.shape[1]] = feature_avhubert
        feature_avhubert = feature_avhubert_padded

        feature_hubert_encoder = feature_hubert_encoder[
            :, : int(lip_len * upsample_hubert)
        ]
        feature_hubert_encoder_padded = np.zeros(
            (feature_hubert_encoder.shape[0], int(lip_len * upsample_hubert))
        )
        feature_hubert_encoder_padded[:, : feature_hubert_encoder.shape[1]] = (
            feature_hubert_encoder
        )
        feature_hubert_encoder = feature_hubert_encoder_padded

        feature_hubert_prj = feature_hubert_prj[:, : int(lip_len * upsample_hubert)]
        feature_hubert_prj_padded = np.zeros(
            (feature_hubert_prj.shape[0], int(lip_len * upsample_hubert))
        )
        feature_hubert_prj_padded[:, : feature_hubert_prj.shape[1]] = feature_hubert_prj
        feature_hubert_prj = feature_hubert_prj_padded

        feature_hubert_cluster = feature_hubert_cluster[
            : int(lip_len * upsample_hubert)
        ]
        feature_hubert_cluster_padded = np.zeros((feature_hubert_cluster.shape[0]))
        feature_hubert_cluster_padded[: feature_hubert_cluster.shape[0]] = (
            feature_hubert_cluster
        )
        feature_hubert_cluster = feature_hubert_cluster_padded

        lip = lip[:lip_len]
        lip_padded = np.zeros((lip_len, 1, 96, 96))
        lip_padded[: lip.shape[0]] = lip
        lip = lip_padded

        wav = torch.from_numpy(wav).to(torch.float32)  # (T,)
        feature = torch.from_numpy(feature).permute(1, 0).to(torch.float32)  # (T, C)
        feature_avhubert = (
            torch.from_numpy(feature_avhubert).permute(1, 0).to(torch.float32)
        )  # (T, C)
        feature_hubert_encoder = torch.from_numpy(feature_hubert_encoder).to(
            torch.float32
        )  # (C, T)
        feature_hubert_prj = torch.from_numpy(feature_hubert_prj).to(
            torch.float32
        )  # (C, T)
        feature_hubert_cluster = torch.from_numpy(feature_hubert_cluster).to(
            torch.float32
        )  # (T,)
        lip = (
            torch.from_numpy(lip).permute(1, 2, 3, 0).to(torch.float32)
        )  # (C, H, W, T)

        lip, feature, feature_avhubert = self.transform(
            wav=wav,
            lip=lip,
            feature=feature,
            feature_avhubert=feature_avhubert,
            lip_mean=self.lip_mean,
            lip_std=self.lip_std,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
        )
        feature_len = torch.tensor(feature.shape[-1])
        feature_hubert_len = torch.tensor(feature_hubert_encoder.shape[-1])
        lip_len = torch.tensor(lip.shape[-1])

        wav = wav.to(torch.float32)
        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feature_avhubert = feature_avhubert.to(torch.float32)
        feature_hubert_encoder = feature_hubert_encoder.to(torch.float32)
        feature_hubert_prj = feature_hubert_prj.to(torch.float32)
        feature_hubert_cluster = feature_hubert_cluster.to(torch.long)
        spk_emb = spk_emb.to(torch.float32)

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
            speaker,
            filename,
        )
