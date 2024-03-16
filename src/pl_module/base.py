import pathlib

import lightning as L
import numpy as np
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from scipy.io.wavfile import write

from src.loss_fn.base import LossFunctions
from src.model.base import BaseModel


class LitBaseModel(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg["training"]["optimizer"]["learning_rate"]
        self.automatic_optimization = True
        self.model = BaseModel(cfg)
        self.loss_fn = LossFunctions()

    def forward(
        self,
        lip: torch.Tensor,
        audio: None,
        lip_len: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.model(
            lip=lip,
            audio=None,
            lip_len=lip_len,
            spk_emb=spk_emb,
        )
        return pred

    def step_typical_process(self, batch: list):
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            filename,
        ) = batch
        pred = self.forward(
            lip=lip,
            audio=None,
            lip_len=lip_len,
            spk_emb=spk_emb,
        )
        loss = self.loss_fn.mae_loss(pred, feature, feature_len, max_len=pred.shape[-1])
        return pred, loss

    def training_step(self, batch: list, batch_index: int):
        pred, loss = self.step_typical_process(batch)
        self.log(
            "train_loss",
            loss,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        return loss

    def validation_step(self, batch: list, batch_index: int):
        pred, loss = self.step_typical_process(batch)
        self.log(
            "val_loss",
            loss,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )

    # def test_step(self, batch: list, batch_index: int):
    #     (
    #         wav,
    #         lip,
    #         feature,
    #         feature_avhubert,
    #         spk_emb,
    #         feature_len,
    #         lip_len,
    #         speaker,
    #         filename,
    #     ) = batch
    #     pred = self.forward(
    #         lip=lip,
    #         audio=None,
    #         lip_len=lip_len,
    #         spk_emb=spk_emb,
    #     )
    #     noise = torch.randn(
    #         pred.shape[0], 1, pred.shape[-1] * self.cfg["data"]["audio"]["hop_length"]
    #     ).to(device=pred.device, dtype=pred.dtype)
    #     wav_pred = self.pwg(noise, pred)
    #     wav_abs = self.pwg(noise, feature)
    #     n_sample_min = min(wav.shape[-1], wav_pred.shape[-1], wav_abs.shape[-1])

    # def save_wav(
    #     self, wav: torch.Tensor, n_sample: int, save_path: pathlib.Path | str
    # ) -> None:
    #     wav = wav.squeeze(0).squeeze(0)
    #     wav = wav.cpu().detach().numpy()
    #     wav /= np.max(np.abs(wav))
    #     wav = wav.astype(np.float32)
    #     wav = wav[:n_sample]
    #     write(str(save_path), rate=self.cfg["data"]["audio"]["sr"], data=wav)

    def configure_optimizers(self):
        optimizer = None
        scheduler = None

        if self.cfg["training"]["optimizer"]["type"] == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.learning_rate,
                betas=(
                    self.cfg["training"]["optimizer"]["beta_1"],
                    self.cfg["training"]["optimizer"]["beta_2"],
                ),
                weight_decay=self.cfg["training"]["optimizer"]["weight_decay"],
            )
        elif self.cfg["training"]["optimizer"]["type"] == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.learning_rate,
                betas=(
                    self.cfg["training"]["optimizer"]["beta_1"],
                    self.cfg["training"]["optimizer"]["beta_2"],
                ),
                weight_decay=self.cfg["training"]["optimizer"]["weight_decay"],
            )
        if optimizer is None:
            raise ValueError("Not supported optimizer")

        if self.cfg["training"]["scheduler"]["type"] == "cawr":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=self.cfg["training"]["params"]["max_epoch"],
                cycle_mult=self.cfg["training"]["scheduler"]["cycle_mult"],
                max_lr=self.cfg["training"]["optimizer"]["learning_rate"],
                min_lr=self.cfg["training"]["scheduler"]["min_lr"],
                warmup_steps=self.cfg["training"]["params"]["max_epoch"]
                * self.cfg["training"]["scheduler"]["warmup_steps"],
                gamma=self.cfg["training"]["scheduler"]["gamma"],
            )
        if scheduler is None:
            raise ValueError("Not supported scheduler")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
