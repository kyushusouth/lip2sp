import logging
import pickle
from pathlib import Path

import omegaconf
import torch
import torch.nn as nn

from src.data_process.utils import get_upsample, get_upsample_speech_ssl
from src.model.utils import load_avhubert

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_channels = cfg.data.audio.n_mels
        self.upsample = get_upsample(cfg)

        self.layers = []
        padding = (cfg.model.decoder.conv.conv_kernel_size - 1) // 2
        for i in range(cfg.model.decoder.conv.n_conv_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=cfg.model.decoder.conv.conv_kernel_size,
                        stride=1,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(cfg.model.decoder.conv.dropout),
                )
            )
        self.layers = nn.ModuleList(self.layers)

        self.out_layer = nn.Linear(
            hidden_channels,
            self.out_channels * self.upsample,
        )

    def forward(self, feature: torch.Tensor):
        """
        args:
            feature: (B, T, C)
        return:
            output: (B, C, T)
        """
        feature = feature.permute(0, 2, 1)  # (B, C, T)
        for layer in self.layers:
            feature = layer(feature)
        feature = feature.permute(0, 2, 1)  # (B, T, C)
        output = self.out_layer(feature)
        output = output.reshape(output.shape[0], -1, self.out_channels).permute(
            0, 2, 1
        )  # (B, C, T)
        return output


class AudioBridingBlock(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.atten = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=hidden_channels // cfg.model.memory_atten.num_channels_per_head,
            dropout=cfg.model.memory_atten.dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, feature, memory, padding_mask):
        """
        args:
            feature: (B, T, C)
            memory: (B, N, C)
            padding_mask: (B, T)
        returns:
            output: (B, T, C)
        """
        output, atten_weights = self.atten(
            query=feature,
            key=memory,
            value=memory,
        )
        output = self.layer_norm(output + feature)
        return output


class AudioBridingModule(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        with open(
            str(
                Path(cfg.path.kablab.hubert.kmeans_dir).expanduser()
                / str(cfg.model.decoder.speech_ssl.layer_index_cluster)
                / f"{cfg.model.decoder.speech_ssl.n_clusters}.pickle"
            ),
            mode="rb",
        ) as f:
            kmeans = pickle.load(f)
        self.memory = torch.from_numpy(kmeans.cluster_centers_).cuda()  # (N, C)

        self.layers = []
        for _ in range(cfg.model.memory_atten.num_layers):
            self.layers.append(AudioBridingBlock(cfg, hidden_channels))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, feature, padding_mask):
        """
        args:
            feature: (B, T, C)
            padding_mask: (B, T)
        returns:
            feature: (B, T, C)
        """
        memory = self.memory.unsqueeze(0).expand(feature.shape[0], -1, -1)
        for layer in self.layers:
            feature = layer(feature, memory, padding_mask)
        return feature


class SpeechMemoryModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.avhubert = load_avhubert(cfg)
        hidden_channels = self.avhubert.encoder_embed_dim

        if cfg.model.memory_atten.use:
            self.abm = AudioBridingModule(cfg, hidden_channels)

        if cfg.model.spk_emb_layer.use:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg.model.spk_emb_layer.dim,
                hidden_channels,
            )

        self.mel_decoder = ConvDecoder(cfg, hidden_channels)
        self.ssl_feature_cluster_decoder_linear = nn.Linear(
            hidden_channels,
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg),
        )

    def extract_feature_avhubert(
        self,
        lip: torch.Tensor,
        padding_mask: torch.Tensor | None,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            padding_mask (Padded elements must be True.): (B, T)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        x = self.avhubert(
            video=lip,
            audio=audio,
            return_res_output=False,
            padding_mask=padding_mask,
        )
        return x

    def transform_for_output(self, x: torch.Tensor, out_dim: int) -> torch.Tensor:
        """
        args:
            x: (B, T, C)
        return:
            x: (B, C, T)
        """
        x = x.reshape(x.shape[0], -1, out_dim)
        x = x.permute(0, 2, 1)
        return x

    def forward(
        self,
        lip: torch.Tensor,
        audio: None,
        spk_emb: torch.Tensor,
        padding_mask_lip: torch.Tensor | None,
    ):
        """
        args:
            lip: (B, C, H, W, T)
            spk_emb: (B, C)
            padding_mask_lip: (B, T)
        return:
            pred_*: (B, C, T)
        """
        lip = lip.permute(0, 1, 4, 2, 3)  # (B, C, T, H, W)
        feature = self.extract_feature_avhubert(
            lip, padding_mask_lip, audio
        )  # (B, T, C)

        if self.cfg.model.memory_atten.use:
            feature = self.abm(feature, padding_mask_lip)

        if self.cfg.model.spk_emb_layer.use:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)  # (B, T, C)
            feature_with_spk_emb = torch.cat([feature, spk_emb], dim=-1)
            feature_with_spk_emb = self.spk_emb_layer(feature_with_spk_emb)

        if self.cfg.model.spk_emb_layer.use:
            pred_mel = self.mel_decoder(feature_with_spk_emb)
        else:
            pred_mel = self.mel_decoder(feature)

        pred_ssl_feature_cluster_linear = self.ssl_feature_cluster_decoder_linear(
            feature
        )
        pred_ssl_feature_cluster_linear = self.transform_for_output(
            pred_ssl_feature_cluster_linear,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )

        return pred_mel, pred_ssl_feature_cluster_linear
