import omegaconf
import torch
import torch.nn as nn
from transformers import AutoModel

from src.data_process.utils import get_upsample, get_upsample_hubert


class HuBERTDecoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hubert = AutoModel.from_pretrained(
            cfg.model.decoder.hubert.model_name
        ).encoder
        self.out_layer_cls = nn.Linear(
            hidden_channels, cfg.model.decoder.hubert.n_clusters + 1
        )
        self.out_layer_reg = nn.Linear(hidden_channels, hidden_channels)
        self.out_layer_mel = nn.Linear(
            hidden_channels,
            cfg.data.audio.n_mels * get_upsample(cfg) // get_upsample_hubert(cfg),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        args:
            x: (B, T, C)
        return:
            output_*: (B, C, T)
        """
        x = self.hubert(x).last_hidden_state
        output_reg = self.out_layer_reg(x).permute(0, 2, 1)
        output_cls = self.out_layer_cls(x).permute(0, 2, 1)
        output_mel = (
            self.out_layer_mel(x)
            .reshape(x.shape[0], -1, self.cfg.data.audio.n_mels)
            .permute(0, 2, 1)
        )
        return output_reg, output_cls, output_mel
