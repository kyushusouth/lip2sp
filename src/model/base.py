import omegaconf
import torch
import torch.nn as nn

from src.model.components.conv_decoder import ConvDecoder
from src.model.utils import load_avhubert, load_raven, load_vatlm


class BaseModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg["model"]["ssl_type"] == "avhubert":
            self.avhubert = load_avhubert(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
        elif cfg["model"]["ssl_type"] == "raven":
            self.raven = load_raven(cfg)
            hidden_channels = self.raven.attention_dim
        elif cfg["model"]["ssl_type"] == "vatlm":
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.vatlm.encoder_embed_dim
        elif cfg["model"]["ssl_type"] == "ensemble":
            self.avhubert = load_avhubert(cfg)
            self.raven = load_raven(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim
                + self.raven.attention_dim
                + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_avhubert_vatlm":
            self.avhubert = load_avhubert(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_avhubert_raven":
            self.avhubert = load_avhubert(cfg)
            self.raven = load_raven(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim + self.raven.attention_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_raven_vatlm":
            self.raven = load_raven(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.vatlm.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.raven.attention_dim + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )

        if cfg["model"]["use_spk_emb"]:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg["model"]["spk_emb_dim"],
                hidden_channels,
            )

        self.decoder = ConvDecoder(cfg, hidden_channels)

    def extract_feature_avhubert(
        self,
        lip: torch.Tensor,
        lip_len: torch.Tensor,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            lip_len: (B,)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        padding_mask = (
            torch.arange(lip.shape[2])
            .unsqueeze(0)
            .expand(lip.shape[0], -1)
            .to(device=lip.device)
        )
        padding_mask = padding_mask > lip_len.unsqueeze(-1)
        x = self.avhubert(
            video=lip,
            audio=audio,
            return_res_output=False,
            padding_mask=padding_mask,
        )  # (B, T, C)
        return x

    def extract_feature_raven(
        self,
        lip: torch.Tensor,
        lip_len: torch.Tensor,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            lip_len: (B,)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        padding_mask = (
            torch.arange(lip.shape[2])
            .unsqueeze(0)
            .expand(lip.shape[0], -1)
            .to(device=lip.device)
        )
        padding_mask = padding_mask <= lip_len.unsqueeze(
            -1
        )  # True for unmasked positions
        padding_mask = padding_mask.unsqueeze(1)  # (B, 1, T)
        x, _ = self.raven(
            xs=lip,
            masks=padding_mask,
        )  # (B, T, C)
        return x

    def extract_feature_vatlm(
        self,
        lip: torch.Tensor,
        lip_len: torch.Tensor,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            lip_len: (B,)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        padding_mask = (
            torch.arange(lip.shape[2])
            .unsqueeze(0)
            .expand(lip.shape[0], -1)
            .to(device=lip.device)
        )
        padding_mask = padding_mask > lip_len.unsqueeze(-1)
        x = self.vatlm(
            video=lip,
            audio=audio,
            padding_mask=padding_mask,
        )  # (B, T, C)
        return x

    def forward(
        self,
        lip: torch.Tensor,
        audio: None,
        lip_len: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, H, W, T)
            audio: (B, C, T)
            lip_len: (B,)
            spk_emb: (B, C)
        return:
            output: (B, C, T)
        """
        lip = lip.permute(0, 1, 4, 2, 3)  # (B, C, T, H, W)

        if self.cfg["model"]["ssl_type"] == "avhubert":
            feature = self.extract_feature_avhubert(lip, lip_len, audio)
        elif self.cfg["model"]["ssl_type"] == "raven":
            feature = self.extract_feature_raven(lip, lip_len, audio)
        elif self.cfg["model"]["ssl_type"] == "vatlm":
            feature = self.extract_feature_vatlm(lip, lip_len, audio)
        elif self.cfg["model"]["ssl_type"] == "ensemble":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(lip, lip_len, audio)
                feature_raven = self.extract_feature_raven(lip, lip_len, audio)
                feature_vatlm = self.extract_feature_vatlm(lip, lip_len, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_raven = self.dropout(feature_raven)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat(
                [feature_avhubert, feature_raven, feature_vatlm], dim=-1
            )
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_avhubert_vatlm":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(lip, lip_len, audio)
                feature_vatlm = self.extract_feature_vatlm(lip, lip_len, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat([feature_avhubert, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_avhubert_raven":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(lip, lip_len, audio)
                feature_raven = self.extract_feature_raven(lip, lip_len, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_raven = self.dropout(feature_raven)
            feature = torch.concat([feature_avhubert, feature_raven], dim=-1)
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_raven_vatlm":
            with torch.no_grad():
                feature_raven = self.extract_feature_raven(lip, lip_len, audio)
                feature_vatlm = self.extract_feature_vatlm(lip, lip_len, audio)
            feature_raven = self.dropout(feature_raven)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat([feature_raven, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)

        if self.cfg["model"]["use_spk_emb"]:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)  # (B, T, C)
            feature = torch.cat([feature, spk_emb], dim=-1)
            feature = self.spk_emb_layer(feature)

        output = self.decoder(feature)

        return output
