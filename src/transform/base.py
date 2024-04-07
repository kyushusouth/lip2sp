import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from src.data_process.utils import get_upsample


class BaseTransform:
    def __init__(self, cfg: omegaconf.DictConfig, train_val_test: str) -> None:
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.hflip = T.RandomHorizontalFlip(p=0.5)

    def horizontal_flip(self, lip: torch.Tensor) -> torch.Tensor:
        """
        lip : (T, C, H, W)
        """
        lip = self.hflip(lip)
        return lip

    def random_crop(self, lip: torch.Tensor, center: bool) -> torch.Tensor:
        """
        lip : (T, C, H, W)
        center : 中心を切り取るか否か
        """
        if center:
            top = left = (
                self.cfg["data"]["video"]["imsize"]
                - self.cfg["data"]["video"]["imsize_cropped"]
            ) // 2
        else:
            top = torch.randint(
                0,
                self.cfg["data"]["video"]["imsize"]
                - self.cfg["data"]["video"]["imsize_cropped"],
                (1,),
            )
            left = torch.randint(
                0,
                self.cfg["data"]["video"]["imsize"]
                - self.cfg["data"]["video"]["imsize_cropped"],
                (1,),
            )
        height = width = self.cfg["data"]["video"]["imsize_cropped"]
        lip = T.functional.crop(lip, top, left, height, width)
        return lip

    def normalization(
        self,
        lip: torch.Tensor,
        feature: torch.Tensor,
        feature_avhubert: torch.Tensor,
        lip_mean: torch.Tensor,
        lip_std: torch.Tensor,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
    ) -> tuple:
        """
        lip : (C, H, W, T)
        feature : (C, T)
        feature_avhubert : (T, C)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)  # (C, 1)
        feat_std = feat_std.unsqueeze(-1)  # (C, 1)
        lip = lip / 255.0
        lip = (lip - lip_mean) / lip_std
        feature = (feature - feat_mean) / feat_std
        feature_avhubert = F.layer_norm(
            feature_avhubert, feature_avhubert.shape[1:]
        ).permute(1, 0)  # (C, T)
        return lip, feature, feature_avhubert

    def segment_masking_segmean(self, lip: torch.Tensor) -> torch.Tensor:
        """
        lip : (C, H, W, T)
        """
        C, H, W, T = lip.shape

        # 最初の1秒から削除するセグメントの開始フレームを選択
        mask_start_idx = torch.randint(0, self.cfg["data"]["video"]["fps"], (1,))
        idx = [i for i in range(T)]

        # マスクする系列長を決定
        mask_length = torch.randint(
            0,
            int(
                self.cfg["data"]["video"]["fps"]
                * self.cfg["training"]["params"]["augs"]["time_masking"][
                    "max_masking_sec"
                ]
            ),
            (1,),
        )

        while True:
            mask_seg_idx = idx[mask_start_idx : mask_start_idx + mask_length]
            seg_mean_lip = torch.mean(
                lip[..., idx[mask_start_idx : mask_start_idx + mask_length]].to(
                    torch.float
                ),
                dim=-1,
            ).to(torch.uint8)
            for i in mask_seg_idx:
                lip[..., i] = seg_mean_lip

            # 開始フレームを1秒先に更新
            mask_start_idx += self.cfg["data"]["video"]["fps"]

            # 次の範囲が動画自体の系列長を超えてしまうならループを抜ける
            if mask_start_idx + mask_length - 1 > T:
                break

        return lip

    def stacker(self, feats: torch.Tensor, stack_order: int) -> torch.Tensor:
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feats = feats.numpy()
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
            -1, stack_order * feat_dim
        )
        feats = torch.from_numpy(feats)
        return feats

    def __call__(
        self,
        lip: torch.Tensor,
        feature: torch.Tensor,
        feature_avhubert: torch.Tensor,
        lip_mean: torch.Tensor,
        lip_std: torch.Tensor,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
    ) -> tuple:
        """
        lip : (C, H, W, T)
        feature : (T, C)
        feature_avhubert : (T, C)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        feature = feature.permute(1, 0)  # (C, T)
        lip = lip.permute(3, 0, 1, 2)  # (T, C, H, W)

        if self.train_val_test == "train":
            if lip.shape[-1] != self.cfg["data"]["video"]["imsize_cropped"]:
                if self.cfg["training"]["params"]["augs"]["random_crop"]["use"]:
                    lip = self.random_crop(lip, center=False)
                else:
                    lip = self.random_crop(lip, center=True)
                if self.cfg["training"]["params"]["augs"]["horizontal_flip"]["use"]:
                    lip = self.horizontal_flip(lip)
            lip = lip.permute(1, 2, 3, 0)  # (C, H, W, T)

            if self.cfg["training"]["params"]["augs"]["time_masking"]["use"]:
                lip = self.segment_masking_segmean(lip)
        else:
            if lip.shape[-1] != self.cfg["data"]["video"]["imsize_cropped"]:
                lip = self.random_crop(lip, center=True)
            lip = lip.permute(1, 2, 3, 0)  # (C, H, W, T)

        feature_avhubert = self.stacker(feature_avhubert, get_upsample(self.cfg))

        lip, feature, feature_avhubert = self.normalization(
            lip=lip,
            feature=feature,
            feature_avhubert=feature_avhubert,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feature_avhubert = feature_avhubert.to(torch.float32)
        return lip, feature, feature_avhubert
