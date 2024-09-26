import logging

import omegaconf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

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


class Decoders(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        hidden_channels: int,
        pred_ssl_conv_feature: bool,
    ):
        super().__init__()
        self.cfg = cfg
        self.pred_ssl_conv_feature = pred_ssl_conv_feature

        if cfg.model.spk_emb_layer.use:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg.model.spk_emb_layer.dim,
                hidden_channels,
            )

        self.mel_decoder = ConvDecoder(cfg, hidden_channels, "mel")
        self.ssl_feature_cluster_decoder = nn.Linear(
            hidden_channels,
            (cfg.model.decoder.speech_ssl.n_clusters + 1)
            * get_upsample_speech_ssl(cfg),
        )
        if pred_ssl_conv_feature:
            self.ssl_conv_feature_decoder = ConvDecoder(
                cfg, hidden_channels, "ssl_conv_feature"
            )

    def forward(self, feature: torch.Tensor, spk_emb: torch.Tensor):
        """
        args:
            feature: (B, T, C)
            spk_emb: (B, C)
        returns:
            pred_mel: (B, C, T)
            pred_ssl_conv_feature: (B, C, T)
            pred_ssl_feature_cluster: (B, T, C)
        """
        if self.cfg.model.spk_emb_layer.use:
            spk_emb_expand = spk_emb.unsqueeze(1).expand(
                -1, feature.shape[1], -1
            )  # (B, T, C)
            feature_with_spk_emb = torch.cat([feature, spk_emb_expand], dim=-1)
            feature_with_spk_emb = self.spk_emb_layer(feature_with_spk_emb)
            pred_mel = self.mel_decoder(feature_with_spk_emb)
            if self.pred_ssl_conv_feature:
                pred_ssl_conv_feature = self.ssl_conv_feature_decoder(
                    feature_with_spk_emb
                )
        else:
            pred_mel = self.mel_decoder(feature)
            if self.ssl_conv_feature_decoder:
                pred_ssl_conv_feature = self.ssl_conv_feature_decoder(feature)

        pred_ssl_feature_cluster = self.ssl_feature_cluster_decoder(feature)

        if self.pred_ssl_conv_feature:
            return pred_mel, pred_ssl_feature_cluster, pred_ssl_conv_feature

        return pred_mel, pred_ssl_feature_cluster


class EnsembleEncoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_channels * 2, hidden_channels)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=cfg.model.decoder.ensemble.nhead,
                dim_feedforward=hidden_channels * 4,
                dropout=cfg.model.decoder.ensemble.dropout,
                batch_first=True,
            ),
            num_layers=cfg.model.decoder.ensemble.num_layers,
        )

    def forward(self, feature, feature_speech_ssl, padding_mask):
        """
        args:
            feature: (B, T, C)
            feature_speech_ssl: (B, T, C)
            padding_mask: (B, T)
                Padded elements must be True.
        returns:
            feature_cat: (B, T, C)
        """
        feature_cat = torch.cat([feature, feature_speech_ssl], dim=-1)
        feature_cat = self.linear(feature_cat)
        feature_cat = self.transformer_encoder(
            src=feature_cat,
            src_key_padding_mask=padding_mask,
        )
        return feature_cat


class SpeechSSLEncoder(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig, hidden_channels: int) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.model.decoder.speech_ssl.load_pretrained_weight:
            self.ssl_model_encoder = AutoModel.from_pretrained(
                cfg.model.decoder.speech_ssl.model_name
            ).encoder
        else:
            self.ssl_model_encoder = AutoModel.from_config(
                AutoConfig.from_pretrained(cfg.model.decoder.speech_ssl.model_name)
            ).encoder

        self.conv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, ssl_conv_feature: torch.Tensor, padding_mask: torch.Tensor):
        """
        args:
            ssl_conv_feature: (B, C, T)
            padding_mask: (B, T)
                Padded elements must be False.
        returns:
            output: (B, T, C)
        """
        output = self.ssl_model_encoder(
            hidden_states=ssl_conv_feature.permute(0, 2, 1),
            attention_mask=padding_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        output = self.conv(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output


class BaseHuBERT2Model(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.avhubert = load_avhubert(cfg)
        hidden_channels = self.avhubert.encoder_embed_dim

        self.decoders_avhubert = Decoders(
            cfg, hidden_channels, pred_ssl_conv_feature=True
        )

        self.ssl_model_encoder = SpeechSSLEncoder(cfg, hidden_channels)

        self.decoders_hubert = Decoders(
            cfg,
            hidden_channels,
            pred_ssl_conv_feature=False,
        )

        self.ensemble_encoder = EnsembleEncoder(cfg, hidden_channels)

        self.decoders_ensemble = Decoders(
            cfg,
            hidden_channels,
            pred_ssl_conv_feature=False,
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
        padding_mask_speech_ssl: torch.Tensor | None,
    ):
        """
        args:
            lip: (B, C, H, W, T)
            spk_emb: (B, C)
            padding_mask_lip: (B, T)
            padding_mask_speech_ssl: (B, T)
        return:
            pred_*: (B, C, T)
        """
        lip = lip.permute(0, 1, 4, 2, 3)  # (B, C, T, H, W)

        feature = self.extract_feature_avhubert(lip, padding_mask_lip, audio)

        pred_mel, pred_ssl_feature_cluster, pred_ssl_conv_feature = (
            self.decoders_avhubert(feature, spk_emb)
        )
        pred_ssl_feature_cluster = self.transform_for_output(
            pred_ssl_feature_cluster,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )

        feature_speech_ssl = self.ssl_model_encoder(
            pred_ssl_conv_feature, padding_mask_speech_ssl
        )

        pred_mel_speech_ssl, pred_ssl_feature_cluster_speech_ssl = self.decoders_hubert(
            feature_speech_ssl, spk_emb
        )
        pred_ssl_feature_cluster_speech_ssl = self.transform_for_output(
            pred_ssl_feature_cluster_speech_ssl,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )

        feature_ensemble = self.ensemble_encoder(
            feature, feature_speech_ssl, padding_mask_lip
        )

        pred_mel_ensemble, pred_ssl_feature_cluster_ensemble = self.decoders_ensemble(
            feature_ensemble, spk_emb
        )
        pred_ssl_feature_cluster_ensemble = self.transform_for_output(
            pred_ssl_feature_cluster_ensemble,
            self.cfg.model.decoder.speech_ssl.n_clusters + 1,
        )

        return (
            pred_mel,
            pred_ssl_conv_feature,
            pred_ssl_feature_cluster,
            pred_mel_speech_ssl,
            pred_ssl_feature_cluster_speech_ssl,
            pred_mel_ensemble,
            pred_ssl_feature_cluster_ensemble,
        )
