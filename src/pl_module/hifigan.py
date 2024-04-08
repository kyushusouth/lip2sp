import lightning as L
import omegaconf
import torch

from src.model.hifigan import Generator, MultiScaleDiscriminator


class LitBaseHuBERTModel(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg["training"]["optimizer"]["learning_rate"]
        self.automatic_optimization = True

        self.gen = Generator(cfg)
        self.sdisc = MultiScaleDiscriminator()
        self.pdisc = MultiScaleDiscriminator()

    def forward(
        self, feature_hubert_cluster: torch.Tensor, spk_emb: torch.Tensor
    ) -> torch.Tensor:
        wav_pred = self.gen(feature_hubert_cluster, spk_emb)
        return wav_pred

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            feature_hubert_encoder,
            feature_hubert_prj,
            feature_hubert_cluster,
            spk_emb,
            feature_len,
            feature_hubert_len,
            lip_len,
            speaker_list,
            filename_list,
        ) = batch

        wav_pred = self.gen(feature_hubert_cluster, spk_emb)
        loss = torch.nn.functional.l1_loss(wav, wav_pred)
        return loss

    def validation_step(self, batch: list, batch_index: int) -> None:
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            feature_hubert_encoder,
            feature_hubert_prj,
            feature_hubert_cluster,
            spk_emb,
            feature_len,
            feature_hubert_len,
            lip_len,
            speaker_list,
            filename_list,
        ) = batch

        wav_pred = self.gen(feature_hubert_cluster, spk_emb)
        loss = torch.nn.functional.l1_loss(wav, wav_pred)

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            params=self.generator.parameters(),
            lr=self.learning_rate,
            betas=(
                self.cfg["training"]["optimizer"]["beta_1"],
                self.cfg["training"]["optimizer"]["beta_2"],
            ),
            weight_decay=self.cfg["training"]["optimizer"]["weight_decay"],
        )
        optimizer_d = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(
                self.cfg["training"]["optimizer"]["beta_1"],
                self.cfg["training"]["optimizer"]["beta_2"],
            ),
            weight_decay=self.cfg["training"]["optimizer"]["weight_decay"],
        )
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g,
            gamma=self.cfg["training"]["scheduler"]["gamma"],
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d,
            gamma=self.cfg["training"]["scheduler"]["gamma"],
        )
        return [
            {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
            {
                "optimizer": optimizer_d,
                "lr_scheduler": {
                    "scheduler": scheduler_d,
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        ]
