from pathlib import Path

import lightning as L
import MeCab
import numpy as np
import omegaconf
import pandas as pd
import torch
import wandb
import whisper
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from jiwer import wer
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.data_process.utils import get_upsample, get_upsample_hubert
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel
from src.loss_fn.base import LossFunctions
from src.model.base_hubert import BaseHuBERTModel
from src.pl_module.hifigan import LitHiFiGANModel


class LitBaseHuBERTModel(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.automatic_optimization = True

        self.model = BaseHuBERTModel(cfg)

        if cfg.model.avhubert.freeze:
            for param in self.model.avhubert.parameters():
                param.requires_grad = False

        if cfg.model.spk_emb_layer.freeze:
            for param in self.model.spk_emb_layer.parameters():
                param.requires_grad = False

        if cfg.model.decoder.conv.freeze:
            for param in self.model.conv_decoder.parameters():
                param.requires_grad = False

        if cfg.model.decoder.hubert.freeze:
            for param in self.model.hubert_decoder.parameters():
                param.requires_grad = False

        self.loss_fn = LossFunctions()

        self.train_step_conv_output_mel_loss_list = []
        self.train_step_conv_output_hubert_encoder_loss_list = []
        self.train_step_conv_output_hubert_cluster_loss_list = []
        self.train_step_conv_output_hubert_prj_loss_list = []
        self.train_step_hubert_output_reg_masked_loss_list = []
        self.train_step_hubert_output_reg_unmasked_loss_list = []
        self.train_step_hubert_output_reg_loss_list = []
        self.train_step_hubert_output_cls_masked_loss_list = []
        self.train_step_hubert_output_cls_unmasked_loss_list = []
        self.train_step_hubert_output_cls_loss_list = []
        self.train_step_total_loss_list = []
        self.train_epoch_conv_output_mel_loss_list = []
        self.train_epoch_conv_output_hubert_encoder_loss_list = []
        self.train_epoch_conv_output_hubert_cluster_loss_list = []
        self.train_epoch_conv_output_hubert_prj_loss_list = []
        self.train_epoch_hubert_output_reg_masked_loss_list = []
        self.train_epoch_hubert_output_reg_unmasked_loss_list = []
        self.train_epoch_hubert_output_reg_loss_list = []
        self.train_epoch_hubert_output_cls_masked_loss_list = []
        self.train_epoch_hubert_output_cls_unmasked_loss_list = []
        self.train_epoch_hubert_output_cls_loss_list = []
        self.train_epoch_total_loss_list = []
        self.val_step_conv_output_mel_loss_list = []
        self.val_step_conv_output_hubert_encoder_loss_list = []
        self.val_step_conv_output_hubert_cluster_loss_list = []
        self.val_step_conv_output_hubert_prj_loss_list = []
        self.val_step_hubert_output_reg_masked_loss_list = []
        self.val_step_hubert_output_reg_unmasked_loss_list = []
        self.val_step_hubert_output_reg_loss_list = []
        self.val_step_hubert_output_cls_masked_loss_list = []
        self.val_step_hubert_output_cls_unmasked_loss_list = []
        self.val_step_hubert_output_cls_loss_list = []
        self.val_step_total_loss_list = []
        self.val_epoch_conv_output_mel_loss_list = []
        self.val_epoch_conv_output_hubert_encoder_loss_list = []
        self.val_epoch_conv_output_hubert_cluster_loss_list = []
        self.val_epoch_conv_output_hubert_prj_loss_list = []
        self.val_epoch_hubert_output_reg_masked_loss_list = []
        self.val_epoch_hubert_output_reg_unmasked_loss_list = []
        self.val_epoch_hubert_output_reg_loss_list = []
        self.val_epoch_hubert_output_cls_masked_loss_list = []
        self.val_epoch_hubert_output_cls_unmasked_loss_list = []
        self.val_epoch_hubert_output_cls_loss_list = []
        self.val_epoch_total_loss_list = []

        self.train_mel_example = {"gt": None, "pred": None}
        self.val_mel_example = {"gt": None, "pred": None}

    def convert_loss_nan_to_zero(self, loss):
        if torch.isnan(loss):
            loss = torch.tensor(0.0).to(dtype=torch.float32, device=self.device)
        return loss

    def calc_losses(self, batch: list) -> tuple:
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

        padding_mask_lip = (
            torch.arange(lip.shape[4])
            .unsqueeze(0)
            .repeat(lip.shape[0], 1)
            .to(device=self.device)
        )
        padding_mask_lip = (padding_mask_lip >= lip_len.unsqueeze(1)).to(
            dtype=torch.bool
        )
        padding_mask_feature = padding_mask_lip.repeat_interleave(
            repeats=get_upsample(self.cfg), dim=1
        )
        padding_mask_feature_hubert = padding_mask_lip.repeat_interleave(
            repeats=get_upsample_hubert(self.cfg), dim=1
        )

        (
            conv_output_mel,
            conv_output_hubert_prj,
            conv_output_hubert_encoder,
            conv_output_hubert_cluster,
            hubert_output_reg,
            hubert_output_cls,
            mask_indices,
        ) = self.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            feature_hubert_prj=feature_hubert_prj,
            padding_mask_lip=padding_mask_lip,
            padding_mask_feature_hubert=padding_mask_feature_hubert,
        )

        conv_output_mel_loss = self.loss_fn.l1_loss(
            pred=conv_output_mel, target=feature, mask=padding_mask_feature
        )
        conv_output_hubert_prj_loss = self.loss_fn.l1_loss(
            pred=conv_output_hubert_prj,
            target=feature_hubert_prj,
            mask=padding_mask_feature_hubert,
        )
        conv_output_hubert_encoder_loss = self.loss_fn.l1_loss(
            pred=conv_output_hubert_encoder,
            target=feature_hubert_encoder,
            mask=padding_mask_feature_hubert,
        )
        conv_output_hubert_cluster_loss = torch.nn.functional.cross_entropy(
            input=conv_output_hubert_cluster,
            target=feature_hubert_cluster,
            ignore_index=0,
        )

        hubert_output_reg_masked_loss = self.loss_fn.l1_loss(
            pred=hubert_output_reg,
            target=feature_hubert_encoder,
            mask=(~mask_indices | padding_mask_feature_hubert),
        )
        hubert_output_reg_masked_loss = self.convert_loss_nan_to_zero(
            hubert_output_reg_masked_loss
        )
        hubert_output_reg_unmasked_loss = self.loss_fn.l1_loss(
            pred=hubert_output_reg,
            target=feature_hubert_encoder,
            mask=(mask_indices | padding_mask_feature_hubert),
        )
        hubert_output_reg_unmasked_loss = self.convert_loss_nan_to_zero(
            hubert_output_reg_unmasked_loss
        )
        hubert_output_reg_loss = (
            hubert_output_reg_masked_loss
            * self.cfg.training.loss_weights.hubert_output_reg_masked_loss
        ) + (
            hubert_output_reg_unmasked_loss
            * self.cfg.training.loss_weights.hubert_output_reg_unmasked_loss
        )

        hubert_output_cls_masked_loss = torch.nn.functional.cross_entropy(
            input=hubert_output_cls,
            target=torch.where(
                (~mask_indices | padding_mask_feature_hubert), 0, feature_hubert_cluster
            ),
            ignore_index=0,
        )
        hubert_output_cls_masked_loss = self.convert_loss_nan_to_zero(
            hubert_output_cls_masked_loss
        )
        hubert_output_cls_unmasked_loss = torch.nn.functional.cross_entropy(
            input=hubert_output_cls,
            target=torch.where(
                (mask_indices | padding_mask_feature_hubert), 0, feature_hubert_cluster
            ),
            ignore_index=0,
        )
        hubert_output_cls_unmasked_loss = self.convert_loss_nan_to_zero(
            hubert_output_cls_unmasked_loss
        )
        hubert_output_cls_loss = (
            hubert_output_cls_masked_loss
            * self.cfg.training.loss_weights.hubert_output_cls_masked_loss
        ) + (
            hubert_output_cls_unmasked_loss
            * self.cfg.training.loss_weights.hubert_output_cls_unmasked_loss
        )

        total_loss = (
            (conv_output_mel_loss * self.cfg.training.loss_weights.conv_output_mel_loss)
            + (
                conv_output_hubert_encoder_loss
                * self.cfg.training.loss_weights["conv_output_hubert_encoder_loss"]
            )
            + (
                conv_output_hubert_cluster_loss
                * self.cfg.training.loss_weights["conv_output_hubert_cluster_loss"]
            )
            + (
                conv_output_hubert_prj_loss
                * self.cfg.training.loss_weights.conv_output_hubert_prj_loss
            )
            + (
                hubert_output_reg_loss
                * self.cfg.training.loss_weights.hubert_output_reg_loss
            )
            + (
                hubert_output_cls_loss
                * self.cfg.training.loss_weights.hubert_output_cls_loss
            )
        )
        return (
            conv_output_mel_loss,
            conv_output_hubert_encoder_loss,
            conv_output_hubert_cluster_loss,
            conv_output_hubert_prj_loss,
            hubert_output_reg_masked_loss,
            hubert_output_reg_unmasked_loss,
            hubert_output_reg_loss,
            hubert_output_cls_masked_loss,
            hubert_output_cls_unmasked_loss,
            hubert_output_cls_loss,
            total_loss,
            feature,
            conv_output_mel,
        )

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        (
            conv_output_mel_loss,
            conv_output_hubert_encoder_loss,
            conv_output_hubert_cluster_loss,
            conv_output_hubert_prj_loss,
            hubert_output_reg_masked_loss,
            hubert_output_reg_unmasked_loss,
            hubert_output_reg_loss,
            hubert_output_cls_masked_loss,
            hubert_output_cls_unmasked_loss,
            hubert_output_cls_loss,
            total_loss,
            feature,
            conv_output_mel,
        ) = self.calc_losses(batch)

        self.log(
            "train_conv_output_mel_loss",
            conv_output_mel_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_conv_output_hubert_encoder_loss",
            conv_output_hubert_encoder_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_conv_output_hubert_cluster_loss",
            conv_output_hubert_cluster_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_conv_output_hubert_prj_loss",
            conv_output_hubert_prj_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_reg_masked_loss",
            hubert_output_reg_masked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_reg_unmasked_loss",
            hubert_output_reg_unmasked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_reg_loss",
            hubert_output_reg_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_cls_masked_loss",
            hubert_output_cls_masked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_cls_unmasked_loss",
            hubert_output_cls_unmasked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_hubert_output_cls_loss",
            hubert_output_cls_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.train_step_conv_output_mel_loss_list.append(conv_output_mel_loss.item())
        self.train_step_conv_output_hubert_encoder_loss_list.append(
            conv_output_hubert_encoder_loss.item()
        )
        self.train_step_conv_output_hubert_cluster_loss_list.append(
            conv_output_hubert_cluster_loss.item()
        )
        self.train_step_conv_output_hubert_prj_loss_list.append(
            conv_output_hubert_prj_loss.item()
        )
        self.train_step_hubert_output_reg_masked_loss_list.append(
            hubert_output_reg_masked_loss.item()
        )
        self.train_step_hubert_output_reg_unmasked_loss_list.append(
            hubert_output_reg_unmasked_loss.item()
        )
        self.train_step_hubert_output_reg_loss_list.append(
            hubert_output_reg_loss.item()
        )
        self.train_step_hubert_output_cls_masked_loss_list.append(
            hubert_output_cls_masked_loss.item()
        )
        self.train_step_hubert_output_cls_unmasked_loss_list.append(
            hubert_output_cls_unmasked_loss.item()
        )
        self.train_step_hubert_output_cls_loss_list.append(
            hubert_output_cls_loss.item()
        )
        self.train_step_total_loss_list.append(total_loss.item())

        self.train_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.train_mel_example["pred"] = (
            conv_output_mel[0].cpu().detach().numpy().astype(np.float32)
        )
        return total_loss

    def validation_step(self, batch: list, batch_index: int) -> None:
        (
            conv_output_mel_loss,
            conv_output_hubert_encoder_loss,
            conv_output_hubert_cluster_loss,
            conv_output_hubert_prj_loss,
            hubert_output_reg_masked_loss,
            hubert_output_reg_unmasked_loss,
            hubert_output_reg_loss,
            hubert_output_cls_masked_loss,
            hubert_output_cls_unmasked_loss,
            hubert_output_cls_loss,
            total_loss,
            feature,
            conv_output_mel,
        ) = self.calc_losses(batch)

        self.log(
            "val_conv_output_mel_loss",
            conv_output_mel_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_conv_output_hubert_encoder_loss",
            conv_output_hubert_encoder_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_conv_output_hubert_cluster_loss",
            conv_output_hubert_cluster_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_conv_output_hubert_prj_loss",
            conv_output_hubert_prj_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_reg_masked_loss",
            hubert_output_reg_masked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_reg_unmasked_loss",
            hubert_output_reg_unmasked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_reg_loss",
            hubert_output_reg_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_cls_masked_loss",
            hubert_output_cls_masked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_cls_unmasked_loss",
            hubert_output_cls_unmasked_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_hubert_output_cls_loss",
            hubert_output_cls_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.val_step_conv_output_mel_loss_list.append(conv_output_mel_loss.item())
        self.val_step_conv_output_hubert_encoder_loss_list.append(
            conv_output_hubert_encoder_loss.item()
        )
        self.val_step_conv_output_hubert_cluster_loss_list.append(
            conv_output_hubert_cluster_loss.item()
        )
        self.val_step_conv_output_hubert_prj_loss_list.append(
            conv_output_hubert_prj_loss.item()
        )
        self.val_step_hubert_output_reg_masked_loss_list.append(
            hubert_output_reg_masked_loss.item()
        )
        self.val_step_hubert_output_reg_unmasked_loss_list.append(
            hubert_output_reg_unmasked_loss.item()
        )
        self.val_step_hubert_output_reg_loss_list.append(hubert_output_reg_loss.item())
        self.val_step_hubert_output_cls_masked_loss_list.append(
            hubert_output_cls_masked_loss.item()
        )
        self.val_step_hubert_output_cls_unmasked_loss_list.append(
            hubert_output_cls_unmasked_loss.item()
        )
        self.val_step_hubert_output_cls_loss_list.append(hubert_output_cls_loss.item())
        self.val_step_total_loss_list.append(total_loss.item())

        self.val_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.val_mel_example["pred"] = (
            conv_output_mel[0].cpu().detach().numpy().astype(np.float32)
        )

    def on_validation_epoch_end(self) -> None:
        self.train_epoch_conv_output_mel_loss_list.append(
            np.mean(self.train_step_conv_output_mel_loss_list)
        )
        self.train_epoch_conv_output_hubert_encoder_loss_list.append(
            np.mean(self.train_step_conv_output_hubert_encoder_loss_list)
        )
        self.train_epoch_conv_output_hubert_cluster_loss_list.append(
            np.mean(self.train_step_conv_output_hubert_cluster_loss_list)
        )
        self.train_epoch_conv_output_hubert_prj_loss_list.append(
            np.mean(self.train_step_conv_output_hubert_prj_loss_list)
        )
        self.train_epoch_hubert_output_reg_masked_loss_list.append(
            np.mean(self.train_step_hubert_output_reg_masked_loss_list)
        )
        self.train_epoch_hubert_output_reg_unmasked_loss_list.append(
            np.mean(self.train_step_hubert_output_reg_unmasked_loss_list)
        )
        self.train_epoch_hubert_output_reg_loss_list.append(
            np.mean(self.train_step_hubert_output_reg_loss_list)
        )
        self.train_epoch_hubert_output_cls_masked_loss_list.append(
            np.mean(self.train_step_hubert_output_cls_masked_loss_list)
        )
        self.train_epoch_hubert_output_cls_unmasked_loss_list.append(
            np.mean(self.train_step_hubert_output_cls_unmasked_loss_list)
        )
        self.train_epoch_hubert_output_cls_loss_list.append(
            np.mean(self.train_step_hubert_output_cls_loss_list)
        )
        self.train_epoch_total_loss_list.append(
            np.mean(self.train_step_total_loss_list)
        )
        self.train_step_conv_output_mel_loss_list.clear()
        self.train_step_conv_output_hubert_encoder_loss_list.clear()
        self.train_step_conv_output_hubert_cluster_loss_list.clear()
        self.train_step_conv_output_hubert_prj_loss_list.clear()
        self.train_step_hubert_output_reg_masked_loss_list.clear()
        self.train_step_hubert_output_reg_unmasked_loss_list.clear()
        self.train_step_hubert_output_reg_loss_list.clear()
        self.train_step_hubert_output_cls_masked_loss_list.clear()
        self.train_step_hubert_output_cls_unmasked_loss_list.clear()
        self.train_step_hubert_output_cls_loss_list.clear()
        self.train_step_total_loss_list.clear()

        self.val_epoch_conv_output_mel_loss_list.append(
            np.mean(self.val_step_conv_output_mel_loss_list)
        )
        self.val_epoch_conv_output_hubert_encoder_loss_list.append(
            np.mean(self.val_step_conv_output_hubert_encoder_loss_list)
        )
        self.val_epoch_conv_output_hubert_cluster_loss_list.append(
            np.mean(self.val_step_conv_output_hubert_cluster_loss_list)
        )
        self.val_epoch_conv_output_hubert_prj_loss_list.append(
            np.mean(self.val_step_conv_output_hubert_prj_loss_list)
        )
        self.val_epoch_hubert_output_reg_masked_loss_list.append(
            np.mean(self.val_step_hubert_output_reg_masked_loss_list)
        )
        self.val_epoch_hubert_output_reg_unmasked_loss_list.append(
            np.mean(self.val_step_hubert_output_reg_unmasked_loss_list)
        )
        self.val_epoch_hubert_output_reg_loss_list.append(
            np.mean(self.val_step_hubert_output_reg_loss_list)
        )
        self.val_epoch_hubert_output_cls_masked_loss_list.append(
            np.mean(self.val_step_hubert_output_cls_masked_loss_list)
        )
        self.val_epoch_hubert_output_cls_unmasked_loss_list.append(
            np.mean(self.val_step_hubert_output_cls_unmasked_loss_list)
        )
        self.val_epoch_hubert_output_cls_loss_list.append(
            np.mean(self.val_step_hubert_output_cls_loss_list)
        )
        self.val_epoch_total_loss_list.append(np.mean(self.val_step_total_loss_list))
        self.val_step_conv_output_mel_loss_list.clear()
        self.val_step_conv_output_hubert_encoder_loss_list.clear()
        self.val_step_conv_output_hubert_cluster_loss_list.clear()
        self.val_step_conv_output_hubert_prj_loss_list.clear()
        self.val_step_hubert_output_reg_masked_loss_list.clear()
        self.val_step_hubert_output_reg_unmasked_loss_list.clear()
        self.val_step_hubert_output_reg_loss_list.clear()
        self.val_step_hubert_output_cls_masked_loss_list.clear()
        self.val_step_hubert_output_cls_unmasked_loss_list.clear()
        self.val_step_hubert_output_cls_loss_list.clear()
        self.val_step_total_loss_list.clear()

        save_epoch_loss_plot(
            title="conv_output_mel_loss",
            train_loss_list=self.train_epoch_conv_output_mel_loss_list,
            val_loss_list=self.val_epoch_conv_output_mel_loss_list,
        )
        save_epoch_loss_plot(
            title="conv_output_hubert_encoder_loss",
            train_loss_list=self.train_epoch_conv_output_hubert_encoder_loss_list,
            val_loss_list=self.val_epoch_conv_output_hubert_encoder_loss_list,
        )
        save_epoch_loss_plot(
            title="conv_output_hubert_cluster_loss",
            train_loss_list=self.train_epoch_conv_output_hubert_cluster_loss_list,
            val_loss_list=self.val_epoch_conv_output_hubert_cluster_loss_list,
        )
        save_epoch_loss_plot(
            title="conv_output_hubert_prj_loss",
            train_loss_list=self.train_epoch_conv_output_hubert_prj_loss_list,
            val_loss_list=self.val_epoch_conv_output_hubert_prj_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_reg_masked_loss",
            train_loss_list=self.train_epoch_hubert_output_reg_masked_loss_list,
            val_loss_list=self.val_epoch_hubert_output_reg_masked_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_reg_unmasked_loss",
            train_loss_list=self.train_epoch_hubert_output_reg_unmasked_loss_list,
            val_loss_list=self.val_epoch_hubert_output_reg_unmasked_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_reg_loss",
            train_loss_list=self.train_epoch_hubert_output_reg_loss_list,
            val_loss_list=self.val_epoch_hubert_output_reg_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_cls_masked_loss",
            train_loss_list=self.train_epoch_hubert_output_cls_masked_loss_list,
            val_loss_list=self.val_epoch_hubert_output_cls_masked_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_cls_unmasked_loss",
            train_loss_list=self.train_epoch_hubert_output_cls_unmasked_loss_list,
            val_loss_list=self.val_epoch_hubert_output_cls_unmasked_loss_list,
        )
        save_epoch_loss_plot(
            title="hubert_output_cls_loss",
            train_loss_list=self.train_epoch_hubert_output_cls_loss_list,
            val_loss_list=self.val_epoch_hubert_output_cls_loss_list,
        )
        save_epoch_loss_plot(
            title="total_loss",
            train_loss_list=self.train_epoch_total_loss_list,
            val_loss_list=self.val_epoch_total_loss_list,
        )

        save_mel(
            cfg=self.cfg,
            gt=self.train_mel_example["gt"],
            pred=self.train_mel_example["pred"],
            filename="train",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.val_mel_example["gt"],
            pred=self.val_mel_example["pred"],
            filename="val",
        )

    def test_step(self, batch: list, batch_index: int) -> None:
        (
            wav_gt,
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
            speaker,
            filename,
        ) = batch

        padding_mask_lip = (
            torch.arange(lip.shape[4])
            .unsqueeze(0)
            .repeat(lip.shape[0], 1)
            .to(device=self.device)
        )
        padding_mask_lip = (padding_mask_lip >= lip_len.unsqueeze(1)).to(
            dtype=torch.bool
        )
        padding_mask_feature = padding_mask_lip.repeat_interleave(
            repeats=get_upsample(self.cfg), dim=1
        )
        padding_mask_feature_hubert = padding_mask_lip.repeat_interleave(
            repeats=get_upsample_hubert(self.cfg), dim=1
        )

        (
            conv_output_mel,
            conv_output_hubert_prj,
            conv_output_hubert_encoder,
            conv_output_hubert_cluster,
            hubert_output_reg,
            hubert_output_cls,
            mask_indices,
        ) = self.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            feature_hubert_prj=feature_hubert_prj,
            padding_mask_lip=padding_mask_lip,
            padding_mask_feature_hubert=padding_mask_feature_hubert,
        )

        if self.cfg.model.decoder.vocoder_input_cluster == "conv":
            inputs_dict = self.hifigan.prepare_inputs_dict(
                feature=conv_output_mel,
                feature_hubert_encoder=conv_output_hubert_encoder,
                feature_hubert_cluster=conv_output_hubert_cluster.argmax(dim=1),
            )
        elif self.cfg.model.decoder.vocoder_input_cluster == "hubert":
            inputs_dict = self.hifigan.prepare_inputs_dict(
                feature=conv_output_mel,
                feature_hubert_encoder=hubert_output_reg,
                feature_hubert_cluster=hubert_output_cls.argmax(dim=1),
            )
        wav_pred = self.hifigan.gen(inputs_dict, spk_emb)

        inputs_dict = self.hifigan.prepare_inputs_dict(
            feature=feature,
            feature_hubert_encoder=feature_hubert_encoder,
            feature_hubert_cluster=feature_hubert_cluster,
        )
        wav_abs = self.hifigan.gen(inputs_dict, spk_emb)

        n_sample_min = min(wav_gt.shape[-1], wav_pred.shape[-1], wav_abs.shape[-1])
        wav_gt = self.process_wav(wav_gt, n_sample_min)
        wav_abs = self.process_wav(wav_abs, n_sample_min)
        wav_pred = self.process_wav(wav_pred, n_sample_min)

        pesq_abs = self.wb_pesq_evaluator(wav_abs, wav_gt)
        pesq_pred = self.wb_pesq_evaluator(wav_pred, wav_gt)
        stoi_abs = self.stoi_evaluator(wav_abs, wav_gt)
        stoi_pred = self.stoi_evaluator(wav_pred, wav_gt)
        estoi_abs = self.estoi_evaluator(wav_abs, wav_gt)
        estoi_pred = self.estoi_evaluator(wav_pred, wav_gt)

        utt = None
        for i, row in self.df_utt.iterrows():
            if str(row["utt_num"]) in filename[0]:
                utt = row["text"].replace("。", "").replace("、", "")
                break
        if utt is None:
            raise ValueError("Utterance was not found.")

        utt_recog_gt = (
            self.speech_recognizer.transcribe(
                wav_gt.to(dtype=torch.float32), language="ja"
            )["text"]
            .replace("。", "")
            .replace("、", "")
        )
        utt_recog_abs = (
            self.speech_recognizer.transcribe(
                wav_abs.to(dtype=torch.float32), language="ja"
            )["text"]
            .replace("。", "")
            .replace("、", "")
        )
        utt_recog_pred = (
            self.speech_recognizer.transcribe(
                wav_pred.to(dtype=torch.float32), language="ja"
            )["text"]
            .replace("。", "")
            .replace("、", "")
        )

        utt_parse = self.mecab.parse(utt)
        utt_recog_gt_parse = self.mecab.parse(utt_recog_gt)
        utt_recog_abs_parse = self.mecab.parse(utt_recog_abs)
        utt_recog_pred_parse = self.mecab.parse(utt_recog_pred)

        wer_gt = self.calc_error_rate(utt_parse, utt_recog_gt_parse)
        wer_abs = self.calc_error_rate(utt_parse, utt_recog_abs_parse)
        wer_pred = self.calc_error_rate(utt_parse, utt_recog_pred_parse)

        data = [
            [
                speaker[0],
                filename[0],
                "gt",
                wandb.Audio(wav_gt.cpu(), sample_rate=self.cfg.data.audio.sr),
                None,
                None,
                None,
                wer_gt,
            ],
            [
                speaker[0],
                filename[0],
                "abs",
                wandb.Audio(wav_abs.cpu(), sample_rate=self.cfg.data.audio.sr),
                pesq_abs,
                stoi_abs,
                estoi_abs,
                wer_abs,
            ],
            [
                speaker[0],
                filename[0],
                "pred",
                wandb.Audio(wav_pred.cpu(), sample_rate=self.cfg.data.audio.sr),
                pesq_pred,
                stoi_pred,
                estoi_pred,
                wer_pred,
            ],
        ]
        self.test_data_list += data

    def on_test_start(self) -> None:
        self.model.eval()

        self.hifigan = LitHiFiGANModel.load_from_checkpoint(
            self.cfg.model.hifigan.model_path,
            cfg=self.cfg,
        )
        self.hifigan.eval()

        self.test_data_columns = [
            "speaker",
            "filename",
            "kind",
            "wav",
            "pesq",
            "stoi",
            "estoi",
            "wer",
        ]
        self.test_data_list = []

        csv_path = Path("~/lip2sp/csv/ATR503.csv").expanduser()
        self.df_utt = pd.read_csv(str(csv_path))
        self.df_utt = self.df_utt.drop(columns=["index"])
        self.df_utt = self.df_utt.loc[self.df_utt["subset"] == "j"]

        self.wb_pesq_evaluator = PerceptualEvaluationSpeechQuality(
            self.cfg.data.audio.sr, "wb"
        )
        self.stoi_evaluator = ShortTimeObjectiveIntelligibility(
            self.cfg.data.audio.sr, extended=False
        )
        self.estoi_evaluator = ShortTimeObjectiveIntelligibility(
            self.cfg.data.audio.sr, extended=True
        )
        self.speech_recognizer = whisper.load_model("large")
        self.mecab = MeCab.Tagger("-Owakati")

    def on_test_end(self) -> None:
        table = wandb.Table(columns=self.test_data_columns, data=self.test_data_list)
        wandb.log({"test_data": table})

        kind_index = self.test_data_columns.index("kind")
        pesq_index = self.test_data_columns.index("pesq")
        stoi_index = self.test_data_columns.index("stoi")
        estoi_index = self.test_data_columns.index("estoi")
        wer_index = self.test_data_columns.index("wer")

        result: dict[str, dict[str, list]] = {
            "gt": {"pesq": [], "stoi": [], "estoi": [], "wer": []},
            "abs": {"pesq": [], "stoi": [], "estoi": [], "wer": []},
            "pred": {"pesq": [], "stoi": [], "estoi": [], "wer": []},
        }

        for test_data in self.test_data_list:
            kind = test_data[kind_index]
            pesq = test_data[pesq_index]
            stoi = test_data[stoi_index]
            estoi = test_data[estoi_index]
            wer = test_data[wer_index]
            result[kind]["pesq"].append(pesq)
            result[kind]["stoi"].append(stoi)
            result[kind]["estoi"].append(estoi)
            result[kind]["wer"].append(wer)

        columns = ["kind", "pesq", "stoi", "estoi", "wer"]
        data_list = []
        for kind, value_dict in result.items():
            if kind == "gt":
                data_list.append(
                    [
                        kind,
                        None,
                        None,
                        None,
                        np.mean(value_dict["wer"]),
                    ]
                )
            else:
                data_list.append(
                    [
                        kind,
                        np.mean(value_dict["pesq"]),
                        np.mean(value_dict["stoi"]),
                        np.mean(value_dict["estoi"]),
                        np.mean(value_dict["wer"]),
                    ]
                )
        table = wandb.Table(columns=columns, data=data_list)
        wandb.log({"metrics_mean": table})

        del (
            self.test_data_columns,
            self.test_data_list,
            self.df_utt,
            self.wb_pesq_evaluator,
            self.stoi_evaluator,
            self.estoi_evaluator,
            self.speech_recognizer,
            self.mecab,
        )

    def process_wav(self, wav: torch.Tensor, n_sample: int) -> torch.Tensor:
        wav = wav.to(torch.float32)
        wav = wav.squeeze(0).squeeze(0)
        wav /= torch.max(torch.abs(wav))
        wav = wav[:n_sample]
        return wav

    def calc_error_rate(self, utt: list, utt_pred: list) -> float:
        wer_gt = None
        try:
            wer_gt = np.clip(wer(utt, utt_pred), a_min=0, a_max=1)
        except:  # noqa: E722
            wer_gt = 1.0
        if wer_gt is None:
            raise ValueError("Word Error Rate was not found.")
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
