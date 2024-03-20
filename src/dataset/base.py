import pathlib
from pathlib import Path

import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from src.data_process.transform import load_data
from src.dataset.utils import get_spk_emb
from src.transform.base import BaseTransform


class BaseDataset(Dataset):
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
        speaker = self.data_path_list[index]["speaker"]
        filename = self.data_path_list[index]["filename"]
        spk_emb = torch.from_numpy(self.embs[speaker])

        wav, feature, feature_avhubert, lip = load_data(
            audio_path=audio_path,
            video_path=video_path,
            cfg=self.cfg,
        )
        wav = torch.from_numpy(wav)
        feature = torch.from_numpy(feature).permute(1, 0)  # (T, C)
        feature_avhubert = torch.from_numpy(feature_avhubert).permute(1, 0)  # (T, C)
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0)  # (C, H, W, T)

        lip, feature, feature_avhubert = self.transform(
            lip=lip,
            feature=feature,
            feature_avhubert=feature_avhubert,
            lip_mean=self.lip_mean,
            lip_std=self.lip_std,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
        )
        feature_len = torch.tensor(feature.shape[-1])
        lip_len = torch.tensor(lip.shape[-1])

        wav = wav.to(torch.float32)
        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feature_avhubert = feature_avhubert.to(torch.float32)
        spk_emb = spk_emb.to(torch.float32)

        return (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            filename,
        )
