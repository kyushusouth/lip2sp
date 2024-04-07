import omegaconf
import torch
import torch.nn as nn
from transformers import AutoModel


class HuBERTDecoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
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
        x = self.hubert(x).last_hidden_state
        output_reg = self.out_layer_reg(x)
        output_cls = self.out_layer_cls(x)
        output_reg = output_reg.permute(0, 2, 1)  # (B, C, T)
        output_cls = output_cls.permute(0, 2, 1)  # (B, C, T)
        return output_reg, output_cls
