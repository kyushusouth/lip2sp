import omegaconf
import torch
import torch.nn as nn

from src.data_process.utils import get_upsample


class ResBlock(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv_layers(x)
        return out + res


class ConvDecoder(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        conv_layers = []
        for i in range(cfg["model"]["decoder"]["n_conv_layers"]):
            conv_layers.append(
                ResBlock(
                    hidden_channels=hidden_channels,
                    kernel_size=cfg["model"]["decoder"]["conv_kernel_size"],
                    dropout=cfg["model"]["decoder"]["dropout"],
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.out_layer = nn.Conv1d(
            hidden_channels,
            cfg["data"]["audio"]["n_mels"] * get_upsample(cfg),
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.out_layer(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = x.reshape(x.shape[0], -1, self.cfg["data"]["audio"]["n_mels"])
        x = x.permute(0, 2, 1)  # (B, C, T)
        return x
