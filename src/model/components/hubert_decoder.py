import omegaconf
import torch
import torch.nn as nn
from transformers import AutoModel

from src.data_process.utils import get_upsample_hubert


class HuBERTDecoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        upsample = get_upsample_hubert(cfg)
        assert upsample == 2
        self.conv_upsample = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=4, stride=upsample, padding=1
        )
        self.hubert = AutoModel.from_pretrained(
            cfg["model"]["decoder"]["hubert"]["model_name"]
        ).encoder
        self.out_layer_cls = nn.Linear(
            hidden_channels, cfg["model"]["decoder"]["hubert"]["n_clusters"] + 1
        )
        self.out_layer_reg = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (B, T, C)
        """
        # hubertは50Hzの入力を必要とするため、25fpsから50fpsにアップサンプリング
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.conv_upsample(x)
        x = x.permute(0, 2, 1)  # (B, T, C)

        x = self.hubert(x).last_hidden_state
        output_reg = self.out_layer_reg(x)
        output_cls = self.out_layer_cls(x)

        output_reg = output_reg.permute(0, 2, 1)  # (B, C, T)
        output_cls = output_cls.permute(0, 2, 1)  # (B, C, T)
        return output_reg, output_cls
