import random

import numpy as np
import omegaconf
import torch

from src.data_process.utils import get_upsample, get_upsample_hubert
from src.pl_module.base_hubert import LitBaseHuBERTModel


class LitBaseHuBERTHiFiGANFineTuning(LitBaseHuBERTModel):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__(cfg=cfg)
        self.automatic_optimization = False

    def train_val_same_process(self, batch: list) -> tuple:
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

        with torch.no_grad():
            (
                conv_output_mel,
                conv_output_hubert_prj,
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

        conv_output_mel_list = []
        hubert_output_reg_list = []
        hubert_output_cls_list = []
        wav_list = []
        for i in range(self.cfg["training"]["batch_size"]):
            input_len_hubert = int(
                self.cfg["training"]["input_sec_hifigan"]
                * self.cfg["data"]["video"]["fps"]
                * get_upsample_hubert(self.cfg)
            )
            start_index_hubert = random.randint(
                0,
                int(lip_len[i] * get_upsample_hubert(self.cfg)) - input_len_hubert,
            )
            input_len_conv_output_mel = int(
                input_len_hubert
                / get_upsample_hubert(self.cfg)
                * get_upsample(self.cfg)
            )
            start_index_conv_output_mel = int(
                start_index_hubert
                / get_upsample_hubert(self.cfg)
                * get_upsample(self.cfg)
            )
            input_len_wav = int(
                input_len_conv_output_mel * self.cfg["data"]["audio"]["hop_length"]
            )
            start_index_wav = int(
                start_index_conv_output_mel * self.cfg["data"]["audio"]["hop_length"]
            )

            conv_output_mel_list.append(
                conv_output_mel[
                    i,
                    :,
                    start_index_conv_output_mel : start_index_conv_output_mel
                    + input_len_conv_output_mel,
                ]
            )
            hubert_output_reg_list.append(
                hubert_output_reg[
                    i,
                    :,
                    start_index_hubert : start_index_hubert + input_len_hubert,
                ]
            )
            hubert_output_cls_list.append(
                hubert_output_cls[
                    i,
                    :,
                    start_index_hubert : start_index_hubert + input_len_hubert,
                ]
            )
            wav_list.append(wav[i, start_index_wav : start_index_wav + input_len_wav])

        conv_output_mel = torch.stack(conv_output_mel_list, dim=0)
        hubert_output_reg = torch.stack(hubert_output_reg_list, dim=0)
        hubert_output_cls = torch.stack(hubert_output_cls_list, dim=0)
        wav = torch.stack(wav_list, dim=0)

        wav = wav.unsqueeze(1)
        mel = self.hifigan.wav2mel(wav)

        inputs_dict = self.hifigan.prepare_inputs_dict(
            feature=conv_output_mel,
            feature_hubert_encoder=hubert_output_reg,
            feature_hubert_cluster=hubert_output_cls,
        )

        wav_pred = self.hifigan.gen(inputs_dict, spk_emb)
        mel_pred = self.hifigan.wav2mel(wav_pred)

        return wav, mel, wav_pred, mel_pred

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        optim_list = self.hifigan.optimizers()
        optim_g = optim_list[0]
        optim_d = optim_list[1]

        wav, mel, wav_pred, mel_pred = self.train_val_same_process(batch)

        self.toggle_optimizer(optim_d)
        y_df_hat_r, y_df_hat_g, _, _ = self.hifigan.mpd(wav, wav_pred.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.hifigan.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        y_ds_hat_r, y_ds_hat_g, _, _ = self.hifigan.msd(wav, wav_pred.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.hifigan.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )
        loss_disc_all = loss_disc_s + loss_disc_f
        self.manual_backward(loss_disc_all)
        optim_d.step()
        optim_d.zero_grad()
        self.untoggle_optimizer(optim_d)

        self.toggle_optimizer(optim_g)
        loss_mel = torch.nn.functional.l1_loss(mel, mel_pred) * 45
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.hifigan.mpd(wav, wav_pred)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.hifigan.msd(wav, wav_pred)
        loss_fm_f = self.hifigan.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.hifigan.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.hifigan.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.hifigan.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        self.manual_backward(loss_gen_all)
        optim_g.step()
        optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)

        self.log(
            "train_loss_disc_f",
            loss_disc_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_disc_s",
            loss_disc_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_disc_all",
            loss_disc_all,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_mel",
            loss_mel,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_fm_f",
            loss_fm_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_fm_s",
            loss_fm_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_gen_f",
            loss_gen_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_gen_s",
            loss_gen_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "train_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.hifigan.train_step_loss_disc_f_list.append(loss_disc_f.item())
        self.hifigan.train_step_loss_disc_s_list.append(loss_disc_s.item())
        self.hifigan.train_step_loss_disc_all_list.append(loss_disc_all.item())
        self.hifigan.train_step_loss_mel_list.append(loss_mel.item())
        self.hifigan.train_step_loss_fm_f_list.append(loss_fm_f.item())
        self.hifigan.train_step_loss_fm_s_list.append(loss_fm_s.item())
        self.hifigan.train_step_loss_gen_f_list.append(loss_gen_f.item())
        self.hifigan.train_step_loss_gen_s_list.append(loss_gen_s.item())
        self.hifigan.train_step_loss_gen_all_list.append(loss_gen_all.item())
        self.hifigan.train_wav_example["gt"] = (
            wav[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )
        self.hifigan.train_wav_example["pred"] = (
            wav_pred[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )

    def validation_step(self, batch: list, batch_index: int) -> None:
        wav, mel, wav_pred, mel_pred = self.train_val_same_process(batch)

        y_df_hat_r, y_df_hat_g, _, _ = self.hifigan.mpd(wav, wav_pred.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.hifigan.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        y_ds_hat_r, y_ds_hat_g, _, _ = self.hifigan.msd(wav, wav_pred.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.hifigan.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )
        loss_disc_all = loss_disc_s + loss_disc_f

        loss_mel = torch.nn.functional.l1_loss(mel, mel_pred) * 45
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.hifigan.mpd(wav, wav_pred)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.hifigan.msd(wav, wav_pred)
        loss_fm_f = self.hifigan.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.hifigan.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.hifigan.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.hifigan.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        self.log(
            "val_loss_disc_f",
            loss_disc_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_disc_s",
            loss_disc_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_disc_all",
            loss_disc_all,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_mel",
            loss_mel,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_fm_f",
            loss_fm_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_fm_s",
            loss_fm_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_gen_f",
            loss_gen_f,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_gen_s",
            loss_gen_s,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.log(
            "val_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg["training"]["batch_size"],
        )
        self.hifigan.val_step_loss_disc_f_list.append(loss_disc_f.item())
        self.hifigan.val_step_loss_disc_s_list.append(loss_disc_s.item())
        self.hifigan.val_step_loss_disc_all_list.append(loss_disc_all.item())
        self.hifigan.val_step_loss_mel_list.append(loss_mel.item())
        self.hifigan.val_step_loss_fm_f_list.append(loss_fm_f.item())
        self.hifigan.val_step_loss_fm_s_list.append(loss_fm_s.item())
        self.hifigan.val_step_loss_gen_f_list.append(loss_gen_f.item())
        self.hifigan.val_step_loss_gen_s_list.append(loss_gen_s.item())
        self.hifigan.val_step_loss_gen_all_list.append(loss_gen_all.item())
        self.hifigan.val_wav_example["gt"] = (
            wav[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )
        self.hifigan.val_wav_example["pred"] = (
            wav_pred[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )

    def on_validation_epoch_end(self) -> None:
        self.hifigan.on_validation_epoch_end()

    def configure_optimizers(self):
        return self.hifigan.configure_optimizers()
