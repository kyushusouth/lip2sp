import random

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torchaudio.transforms import AddNoise
from torchvision.transforms import (
    CenterCrop,
    GaussianBlur,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
)

from src.data_process.utils import get_upsample, wav2mel


class BaseTransform:
    def __init__(self, cfg: omegaconf.DictConfig, train_val_test: str) -> None:
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.random_crop = RandomCrop(
            size=(
                cfg.data.video.imsize_cropped,
                cfg.data.video.imsize_cropped,
            )
        )
        self.center_crop = CenterCrop(
            size=(
                cfg.data.video.imsize_cropped,
                cfg.data.video.imsize_cropped,
            )
        )
        self.horizontal_flip = RandomHorizontalFlip(
            p=cfg.training.augs.horizontal_flip.p
        )
        self.random_erasing = RandomErasing(
            p=cfg.training.augs.random_erasing.p,
            scale=(
                cfg.training.augs.random_erasing.scale_min,
                cfg.training.augs.random_erasing.scale_max,
            ),
            ratio=(
                cfg.training.augs.random_erasing.ratio_min,
                cfg.training.augs.random_erasing.scale_max,
            ),
            value=cfg.training.augs.random_erasing.value,
        )
        self.add_noise = AddNoise()
        self.gaussian_blur = GaussianBlur(
            kernel_size=(
                self.cfg.training.augs.add_gaussian_noise.kernel_size,
                self.cfg.training.augs.add_gaussian_noise.kernel_size,
            ),
            sigma=(
                self.cfg.training.augs.add_gaussian_noise.sigma_min,
                self.cfg.training.augs.add_gaussian_noise.sigma_max,
            ),
        )

    def apply_random_crop(self, lip: torch.Tensor, center: bool) -> torch.Tensor:
        """
        lip : (T, C, H, W)
        center : 中心を切り取るか否か
        """
        if center:
            return self.center_crop(lip)
        else:
            return self.random_crop(lip)

    def apply_horizontal_flip(self, lip: torch.Tensor) -> torch.Tensor:
        """
        lip : (T, C, H, W)
        """
        return self.horizontal_flip(lip)

    def apply_random_erasing(self, lip: torch.Tensor) -> torch.Tensor:
        """
        lip : (T, C, H, W)
        """
        return self.random_erasing(lip)

    def apply_time_masking(self, lip: torch.Tensor) -> torch.Tensor:
        """
        lip : (C, H, W, T)
        毎秒ランダムなフレーム数分だけ、平均化したベクトルに置き換える
        """
        t = 0
        while t < lip.shape[3]:
            mask_length = random.randint(
                0,
                int(
                    self.cfg.data.video.fps
                    * self.cfg.training.augs.time_masking.max_masking_sec
                ),
            )
            mask_start_frame = random.randint(
                t, t + self.cfg.data.video.fps - mask_length
            )
            lip_mask = (
                lip[:, :, :, mask_start_frame : mask_start_frame + mask_length]
                .to(torch.float)
                .mean(dim=3, keepdim=True)
                .to(torch.uint8)
            )
            lip[:, :, :, mask_start_frame : mask_start_frame + mask_length] = lip_mask
            t += self.cfg.data.video.fps

        return lip

    def apply_spec_gaussian_blur(self, feature: torch.Tensor) -> torch.Tensor:
        """
        feature: (C, T)
        """
        feature = feature.unsqueeze(0)
        feature = self.gaussian_blur(feature)
        feature = feature.squeeze(0)
        return feature

    def apply_add_gaussian_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (T,)
        """
        snr = random.randint(
            self.cfg.training.augs.spec_gaussian_blur.snr_min,
            self.cfg.training.augs.spec_gaussian_blur.snr_max,
        )
        wav = self.add_noise(wav, noise=torch.randn(wav.shape), snr=torch.tensor(snr))
        return wav

    def apply_normalization(
        self,
        lip: torch.Tensor,
        feature: torch.Tensor,
        feature_avhubert: torch.Tensor,
        lip_mean: torch.Tensor,
        lip_std: torch.Tensor,
    ) -> tuple:
        """
        lip : (C, H, W, T)
        feature : (C, T)
        feature_avhubert : (T, C)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1, 1)
        lip = lip / 255.0
        lip = (lip - lip_mean) / lip_std

        feature_avhubert = F.layer_norm(
            feature_avhubert, feature_avhubert.shape[1:]
        ).permute(1, 0)  # (C, T)
        return lip, feature, feature_avhubert

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
        wav: torch.Tensor,
        lip: torch.Tensor,
        feature: torch.Tensor,
        feature_avhubert: torch.Tensor,
        lip_mean: torch.Tensor,
        lip_std: torch.Tensor,
    ) -> tuple:
        """
        wav: (T,)
        lip : (C, H, W, T)
        feature : (T, C)
        feature_avhubert : (T, C)
        lip_mean, lip_std : (C,)
        """
        feature = feature.permute(1, 0)  # (C, T)
        lip = lip.permute(3, 0, 1, 2)  # (T, C, H, W)

        if self.train_val_test == "train":
            if self.cfg.training.augs.random_crop.use:
                lip = self.apply_random_crop(lip, center=False)
            else:
                lip = self.apply_random_crop(lip, center=True)
            if self.cfg.training.augs.horizontal_flip.use:
                lip = self.apply_horizontal_flip(lip)

            lip = lip.permute(1, 2, 3, 0)  # (C, H, W, T)

            if self.cfg.training.augs.time_masking.use:
                lip = self.apply_time_masking(lip)

            if self.cfg.training.augs.add_gaussian_noise.use:
                feature = torch.from_numpy(
                    wav2mel(
                        self.apply_add_gaussian_noise(wav).numpy(),
                        self.cfg,
                        ref_max=False,
                    )
                )[:, : feature.shape[1]]

            if self.cfg.training.augs.spec_gaussian_blur.use:
                feature = self.apply_spec_gaussian_blur(feature)
        else:
            lip = self.apply_random_crop(lip, center=True)
            lip = lip.permute(1, 2, 3, 0)  # (C, H, W, T)

        feature_avhubert = self.stacker(feature_avhubert, get_upsample(self.cfg))

        lip, feature, feature_avhubert = self.apply_normalization(
            lip=lip,
            feature=feature,
            feature_avhubert=feature_avhubert,
            lip_mean=lip_mean,
            lip_std=lip_std,
        )

        if self.train_val_test == "train":
            if self.cfg.training.augs.random_erasing.use:
                lip = lip.permute(3, 0, 1, 2)  # (T, C, H, W)
                lip = self.apply_random_erasing(lip)
                lip = lip.permute(1, 2, 3, 0)  # (C, H, W, T)

        return lip, feature, feature_avhubert
