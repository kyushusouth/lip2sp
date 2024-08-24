import random

import lightning as L
import numpy as np
import omegaconf
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from jiwer import wer

import wandb
from src.dataset.utils import get_token_index_mapping
from src.log_fn.save_loss import save_epoch_loss_plot
from src.model.layerwise_asr import LayerwiseASRModel


class LayerwiseASRModule(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.automatic_optimization = True

        self.model = LayerwiseASRModel(cfg)
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.ctc_loss_func = torch.nn.CTCLoss(blank=cfg.training.token_index.blank)

        self.train_step_ctc_loss_list = []
        self.train_epoch_ctc_loss_list = []
        self.val_step_ctc_loss_list = []
        self.val_epoch_ctc_loss_list = []

    def calc_losses(self, batch: list) -> torch.Tensor:
        (
            wav,
            wav_len,
            feature_len,
            token,
            token_len,
            speaker,
            filename,
        ) = batch

        padding_mask = (
            torch.arange(wav.shape[1]).unsqueeze(0).repeat(wav.shape[0], 1).cuda()
        )
        padding_mask = padding_mask < wav_len.unsqueeze(1)

        output = self.model(
            x=wav,
            feature_len=feature_len,
            padding_mask=padding_mask,
        )
        output = output.permute(1, 0, 2).log_softmax(dim=2)  # (T, B, C)

        feature_len = torch.clamp(feature_len, max=output.shape[0])

        ctc_loss = self.ctc_loss_func(output, token, feature_len, token_len)
        return ctc_loss

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        ctc_loss = self.calc_losses(batch)
        self.log(
            "train_ctc_loss",
            ctc_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.train_step_ctc_loss_list.append(ctc_loss.item())
        return ctc_loss

    def validation_step(self, batch: list, batch_index: int) -> None:
        ctc_loss = self.calc_losses(batch)
        self.log(
            "val_ctc_loss",
            ctc_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.val_step_ctc_loss_list.append(ctc_loss.item())

    def on_validation_epoch_end(self) -> None:
        self.train_epoch_ctc_loss_list.append(np.mean(self.train_step_ctc_loss_list))
        self.train_step_ctc_loss_list.clear()
        self.val_epoch_ctc_loss_list.append(np.mean(self.val_step_ctc_loss_list))
        self.val_step_ctc_loss_list.clear()

        save_epoch_loss_plot(
            title="ctc_loss",
            train_loss_list=self.train_epoch_ctc_loss_list,
            val_loss_list=self.val_epoch_ctc_loss_list,
        )

    def test_step(self, batch: list, batch_index: int) -> None:
        (
            wav,
            wav_len,
            feature_len,
            token,
            token_len,
            speaker,
            filename,
        ) = batch

        padding_mask = (
            torch.arange(wav.shape[1]).unsqueeze(0).repeat(wav.shape[0], 1).cuda()
        )
        padding_mask = padding_mask < wav_len.unsqueeze(1)

        output = self.model(
            x=wav,
            feature_len=feature_len,
            padding_mask=padding_mask,
        )
        output = output.permute(1, 0, 2).log_softmax(dim=2)  # (T, B, C)
        output = torch.argmax(output, dim=2)  # (T, B)

        token = token.cpu()
        output = output.cpu()
        token_gt = " ".join([self.id_to_token[t.item()] for t in token[0]])
        token_pred = " ".join([self.id_to_token[o.item()] for o in output[:, 0]])
        error_rate = self.calc_error_rate(token_gt, token_pred)

        data = [
            [
                speaker[0],
                filename[0],
                error_rate,
                token_gt,
                token_pred,
            ]
        ]
        self.test_data_list += data

    def on_test_start(self) -> None:
        self.token_to_id, self.id_to_token = get_token_index_mapping(self.cfg)

        self.test_data_columns = [
            "speaker",
            "filename",
            "wer",
            "gt",
            "pred",
        ]
        self.test_data_list = []

    def on_test_end(self) -> None:
        table = wandb.Table(columns=self.test_data_columns, data=self.test_data_list)
        wandb.log({"test_data": table})
        wandb.log(
            {"error_rate_mean": np.mean([data[2] for data in self.test_data_list])}
        )

    def calc_error_rate(self, utt: str, utt_pred: str) -> float:
        wer_gt = None
        try:
            wer_gt = np.clip(wer(utt, utt_pred), a_min=0, a_max=1)
        except:  # noqa: E722
            wer_gt = 1.0
        return wer_gt

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            betas=(
                self.cfg.training.optimizer.beta_1,
                self.cfg.training.optimizer.beta_2,
            ),
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.cfg.training.max_epoch,
            cycle_mult=self.cfg.training.scheduler.cycle_mult,
            max_lr=self.cfg.training.optimizer.learning_rate,
            min_lr=self.cfg.training.scheduler.min_lr,
            warmup_steps=self.cfg.training.max_epoch
            * self.cfg.training.scheduler.warmup_steps,
            gamma=self.cfg.training.scheduler.gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
