import logging

import omegaconf
import torch
import torch.nn as nn
from transformers import AutoModel

from src.data_process.utils import get_upsample, get_upsample_speech_ssl
from src.model.utils import load_avhubert

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        hidden_channels: int,
        output_type: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if output_type == "mel":
            self.out_channels = cfg.data.audio.n_mels
            self.upsample = get_upsample(cfg)
        elif output_type == "ssl_conv_feature":
            self.out_channels = cfg.model.decoder.speech_ssl.conv_output_dim
            self.upsample = get_upsample_speech_ssl(cfg)

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


class SSLModelDecoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.ssl_model_encoder = AutoModel.from_pretrained(
            cfg.model.decoder.speech_ssl.model_name
        ).encoder
        self.out_layer = nn.Linear(
            hidden_channels,
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg),
        )

    def forward(self, feature: torch.Tensor, padding_mask: torch.Tensor):
        """
        args:
            feature: (B, T, C)
            padding_mask: (B, T)
                パディング部をFalseとするbool型
        returns:
            output: (B, T, C)
        """
        feature = self.ssl_model_encoder(
            hidden_states=feature,
            attention_mask=padding_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state  # (B, T, C)
        output = self.out_layer(feature)
        return output


class EnsembleDecoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.first_layer = nn.Linear(
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg)
            * 2,
            hidden_channels,
        )

        self.hidden_layers = []
        for i in range(cfg.model.decoder.ensemble.n_hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(cfg.model.decoder.ensemble.dropout),
                )
            )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.out_layer = nn.Linear(
            hidden_channels,
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg),
        )

    def forward(self, x):
        """
        args:
            x: (B, T, C)
        returns:
            x: (B, T, C)
        """
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class WithSpeechSSLModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.avhubert = load_avhubert(cfg)
        hidden_channels = self.avhubert.encoder_embed_dim

        if cfg.model.spk_emb_layer.use:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg.model.spk_emb_layer.dim,
                hidden_channels,
            )

        self.mel_decoder = ConvDecoder(cfg, hidden_channels, "mel")
        self.ssl_conv_feature_decoder = ConvDecoder(
            cfg, hidden_channels, "ssl_conv_feature"
        )
        self.ssl_feature_cluster_decoder_linear = nn.Linear(
            hidden_channels,
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg),
        )
        self.ssl_feature_cluster_decoder_ssl = SSLModelDecoder(cfg, hidden_channels)
        self.ssl_feature_cluster_decoder_ensemble = EnsembleDecoder(
            cfg, hidden_channels
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
        padding_mask_feature: torch.Tensor | None,
    ):
        """
        args:
            lip: (B, C, H, W, T)
            spk_emb: (B, C)
            padding_mask_lip: (B, T)
            padding_mask_feature: (B, T)
        return:
            pred_*: (B, C, T)
        """
        lip = lip.permute(0, 1, 4, 2, 3)  # (B, C, T, H, W)
        feature = self.extract_feature_avhubert(lip, padding_mask_lip, audio)

        if self.cfg.model.spk_emb_layer.use:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)  # (B, T, C)
            feature_with_spk_emb = torch.cat([feature, spk_emb], dim=-1)
            feature_with_spk_emb = self.spk_emb_layer(feature_with_spk_emb)

        if self.cfg.model.spk_emb_layer.use:
            pred_mel = self.mel_decoder(feature_with_spk_emb)
            pred_ssl_conv_feature = self.ssl_conv_feature_decoder(feature_with_spk_emb)
        else:
            pred_mel = self.mel_decoder(feature)
            pred_ssl_conv_feature = self.ssl_conv_feature_decoder(feature)

        pred_ssl_feature_cluster_linear = self.ssl_feature_cluster_decoder_linear(
            feature
        )
        pred_ssl_feature_cluster_ssl = self.ssl_feature_cluster_decoder_ssl(
            feature, padding_mask_feature
        )
        pred_ssl_feature_cluster_ensemble = self.ssl_feature_cluster_decoder_ensemble(
            torch.cat(
                [pred_ssl_feature_cluster_linear, pred_ssl_feature_cluster_ssl], dim=2
            )
        )

        pred_ssl_feature_cluster_linear = self.transform_for_output(
            pred_ssl_feature_cluster_linear,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )
        pred_ssl_feature_cluster_ssl = self.transform_for_output(
            pred_ssl_feature_cluster_ssl,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )
        pred_ssl_feature_cluster_ensemble = self.transform_for_output(
            pred_ssl_feature_cluster_ensemble,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )

        return (
            pred_mel,
            pred_ssl_conv_feature,
            pred_ssl_feature_cluster_linear,
            pred_ssl_feature_cluster_ssl,
            pred_ssl_feature_cluster_ensemble,
        )
