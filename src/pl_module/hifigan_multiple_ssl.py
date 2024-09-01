import itertools
import logging
import random
from pathlib import Path

import librosa
import lightning as L
import MeCab
import numpy as np
import omegaconf
import pandas as pd
import torch
import whisper
from jiwer import wer
from scipy.io.wavfile import write
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

import wandb
from src.data_process.utils import get_upsample, get_upsample_speech_ssl, wav2mel
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel, save_wav_table
from src.model.hifigan_multiplt_ssl import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from src.pl_module.with_speech_ssl import LitWithSpeechSSLModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LitHiFiGANMultipltSSLModel(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.automatic_optimization = False

        self.gen = Generator(cfg)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        if cfg.model.hifigan.freeze:
            logger.info("freeze hifigan parameters.")
            for param in self.gen.parameters():
                param.requires_grad = False
            for param in self.mpd.parameters():
                param.requires_grad = False
            for param in self.msd.parameters():
                param.requires_grad = False

        self.mel_basis: None | torch.Tensor = None
        self.hann_window: None | torch.Tensor = None

        self.train_step_loss_disc_f_list: list[float] = []
        self.train_step_loss_disc_s_list: list[float] = []
        self.train_step_loss_disc_all_list: list[float] = []
        self.train_step_loss_mel_list: list[float] = []
        self.train_step_loss_fm_f_list: list[float] = []
        self.train_step_loss_fm_s_list: list[float] = []
        self.train_step_loss_gen_f_list: list[float] = []
        self.train_step_loss_gen_s_list: list[float] = []
        self.train_step_loss_gen_all_list: list[float] = []
        self.val_step_loss_disc_f_list: list[float] = []
        self.val_step_loss_disc_s_list: list[float] = []
        self.val_step_loss_disc_all_list: list[float] = []
        self.val_step_loss_mel_list: list[float] = []
        self.val_step_loss_fm_f_list: list[float] = []
        self.val_step_loss_fm_s_list: list[float] = []
        self.val_step_loss_gen_f_list: list[float] = []
        self.val_step_loss_gen_s_list: list[float] = []
        self.val_step_loss_gen_all_list: list[float] = []
        self.train_epoch_loss_disc_f_list: list[float] = []
        self.train_epoch_loss_disc_s_list: list[float] = []
        self.train_epoch_loss_disc_all_list: list[float] = []
        self.train_epoch_loss_mel_list: list[float] = []
        self.train_epoch_loss_fm_f_list: list[float] = []
        self.train_epoch_loss_fm_s_list: list[float] = []
        self.train_epoch_loss_gen_f_list: list[float] = []
        self.train_epoch_loss_gen_s_list: list[float] = []
        self.train_epoch_loss_gen_all_list: list[float] = []
        self.val_epoch_loss_disc_f_list: list[float] = []
        self.val_epoch_loss_disc_s_list: list[float] = []
        self.val_epoch_loss_disc_all_list: list[float] = []
        self.val_epoch_loss_mel_list: list[float] = []
        self.val_epoch_loss_fm_f_list: list[float] = []
        self.val_epoch_loss_fm_s_list: list[float] = []
        self.val_epoch_loss_gen_f_list: list[float] = []
        self.val_epoch_loss_gen_s_list: list[float] = []
        self.val_epoch_loss_gen_all_list: list[float] = []
        self.train_wav_example: dict[str, np.ndarray] = {
            "gt": np.random.rand(1),
            "pred": np.random.rand(1),
        }
        self.val_wav_example: dict[str, np.ndarray] = {
            "gt": np.random.rand(1),
            "pred": np.random.rand(1),
        }

    def dynamic_range_compression_torch(
        self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
    ) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def wav2mel(self, wav: torch.Tensor) -> torch.Tensor:
        if self.mel_basis is None and self.hann_window is None:
            mel_filterbank = librosa.filters.mel(
                sr=self.cfg.data.audio.sr,
                n_fft=self.cfg.model.hifigan.loss.n_fft,
                n_mels=self.cfg.model.hifigan.loss.n_mels,
                fmin=self.cfg.data.audio.f_min,
                fmax=self.cfg.data.audio.f_max,
            )
            self.mel_basis = torch.from_numpy(mel_filterbank).to(
                dtype=torch.float32, device=self.device
            )
            self.hann_window = torch.hann_window(
                self.cfg.model.hifigan.loss.win_length
            ).to(device=self.device)

        wav = torch.nn.functional.pad(
            wav,
            (
                int(
                    (
                        self.cfg.model.hifigan.loss.n_fft
                        - self.cfg.model.hifigan.loss.hop_length
                    )
                    / 2
                ),
                int(
                    (
                        self.cfg.model.hifigan.loss.n_fft
                        - self.cfg.model.hifigan.loss.hop_length
                    )
                    / 2
                ),
            ),
            mode="reflect",
        )
        wav = wav.squeeze(1)

        spec = torch.stft(
            wav,
            self.cfg.model.hifigan.loss.n_fft,
            hop_length=self.cfg.model.hifigan.loss.hop_length,
            win_length=self.cfg.model.hifigan.loss.win_length,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        if self.mel_basis is None:
            raise ValueError("self.mel_basis is None")
        spec = torch.matmul(self.mel_basis, spec)
        spec = self.dynamic_range_compression_torch(spec)
        return spec

    def feature_loss(self, fmap_r: torch.Tensor, fmap_g: torch.Tensor) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    def discriminator_loss(
        self, disc_real_outputs: torch.Tensor, disc_generated_outputs: torch.Tensor
    ) -> tuple:
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses

    def generator_loss(self, disc_outputs: torch.Tensor) -> tuple:
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l
        return loss, gen_losses

    def prepare_inputs_dict(
        self,
        mel: torch.Tensor | None,
        hubert_final_feature_cluster: torch.Tensor | None,
        wav2vec2_final_feature_cluster: torch.Tensor | None,
        data2vec_final_feature_cluster: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        """
        args:
            mel: (B, C, T)
            hubert_final_feature_cluster: (B, T)
            wav2vec2_final_feature_cluster: (B, T)
            data2vec_final_feature_cluster: (B, T)
        """
        if mel is not None:
            mel = mel.permute(0, 2, 1)
            mel = mel.reshape(mel.shape[0], -1, int(self.cfg.data.audio.n_mels * 2))

        inputs_dict = {
            "mel": mel,
            "hubert_final_feature_cluster": hubert_final_feature_cluster,
            "wav2vec2_final_feature_cluster": wav2vec2_final_feature_cluster,
            "data2vec_final_feature_cluster": data2vec_final_feature_cluster,
        }
        return inputs_dict

    def calc_losses(self, batch: list) -> tuple:
        (
            wav,
            lip,
            feature,
            hubert_conv_feature,
            hubert_final_feature,
            hubert_final_feature_cluster,
            wav2vec2_conv_feature,
            wav2vec2_final_feature,
            wav2vec2_final_feature_cluster,
            data2vec_conv_feature,
            data2vec_final_feature,
            data2vec_final_feature_cluster,
            spk_emb,
            feature_len,
            feature_ssl_len,
            lip_len,
            speaker,
            filename,
        ) = batch
        wav = wav.unsqueeze(1)
        mel = self.wav2mel(wav)

        inputs_dict = self.prepare_inputs_dict(
            mel=feature,
            hubert_final_feature_cluster=hubert_final_feature_cluster,
            wav2vec2_final_feature_cluster=wav2vec2_final_feature_cluster,
            data2vec_final_feature_cluster=data2vec_final_feature_cluster,
        )

        wav_pred = self.gen(inputs_dict, spk_emb)
        mel_pred = self.wav2mel(wav_pred)

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(wav, wav_pred.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(wav, wav_pred.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )
        loss_disc_all = loss_disc_s + loss_disc_f

        loss_mel = torch.nn.functional.l1_loss(mel, mel_pred) * 45
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(wav, wav_pred)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(wav, wav_pred)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        return (
            wav,
            wav_pred,
            loss_disc_f,
            loss_disc_s,
            loss_disc_all,
            loss_mel,
            loss_fm_f,
            loss_fm_s,
            loss_gen_f,
            loss_gen_s,
            loss_gen_all,
        )

    def training_step(self, batch: list, batch_index: int) -> None:
        optim_list = self.optimizers()
        optim_g = optim_list[0]
        optim_d = optim_list[1]

        (
            wav,
            wav_pred,
            loss_disc_f,
            loss_disc_s,
            loss_disc_all,
            loss_mel,
            loss_fm_f,
            loss_fm_s,
            loss_gen_f,
            loss_gen_s,
            loss_gen_all,
        ) = self.calc_losses(batch)

        self.toggle_optimizer(optim_d)
        self.manual_backward(loss_disc_all / self.cfg.training.accumulate_grad_batches)
        if (batch_index + 1) % self.cfg.training.accumulate_grad_batches == 0:
            optim_d.step()
            optim_d.zero_grad()
        self.untoggle_optimizer(optim_d)

        self.toggle_optimizer(optim_g)
        self.manual_backward(loss_gen_all / self.cfg.training.accumulate_grad_batches)
        if (batch_index + 1) % self.cfg.training.accumulate_grad_batches == 0:
            optim_g.step()
            optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)

        self.log(
            "train_loss_disc_f",
            loss_disc_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_disc_s",
            loss_disc_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_disc_all",
            loss_disc_all,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_mel",
            loss_mel,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_fm_f",
            loss_fm_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_fm_s",
            loss_fm_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_gen_f",
            loss_gen_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_gen_s",
            loss_gen_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.train_step_loss_disc_f_list.append(loss_disc_f.item())
        self.train_step_loss_disc_s_list.append(loss_disc_s.item())
        self.train_step_loss_disc_all_list.append(loss_disc_all.item())
        self.train_step_loss_mel_list.append(loss_mel.item())
        self.train_step_loss_fm_f_list.append(loss_fm_f.item())
        self.train_step_loss_fm_s_list.append(loss_fm_s.item())
        self.train_step_loss_gen_f_list.append(loss_gen_f.item())
        self.train_step_loss_gen_s_list.append(loss_gen_s.item())
        self.train_step_loss_gen_all_list.append(loss_gen_all.item())
        self.train_wav_example["gt"] = (
            wav[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )
        self.train_wav_example["pred"] = (
            wav_pred[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )

    def validation_step(self, batch: list, batch_index: int) -> None:
        (
            wav,
            wav_pred,
            loss_disc_f,
            loss_disc_s,
            loss_disc_all,
            loss_mel,
            loss_fm_f,
            loss_fm_s,
            loss_gen_f,
            loss_gen_s,
            loss_gen_all,
        ) = self.calc_losses(batch)

        self.log(
            "val_loss_disc_f",
            loss_disc_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_disc_s",
            loss_disc_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_disc_all",
            loss_disc_all,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_mel",
            loss_mel,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_fm_f",
            loss_fm_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_fm_s",
            loss_fm_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_gen_f",
            loss_gen_f,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_gen_s",
            loss_gen_s,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.val_step_loss_disc_f_list.append(loss_disc_f.item())
        self.val_step_loss_disc_s_list.append(loss_disc_s.item())
        self.val_step_loss_disc_all_list.append(loss_disc_all.item())
        self.val_step_loss_mel_list.append(loss_mel.item())
        self.val_step_loss_fm_f_list.append(loss_fm_f.item())
        self.val_step_loss_fm_s_list.append(loss_fm_s.item())
        self.val_step_loss_gen_f_list.append(loss_gen_f.item())
        self.val_step_loss_gen_s_list.append(loss_gen_s.item())
        self.val_step_loss_gen_all_list.append(loss_gen_all.item())
        self.val_wav_example["gt"] = (
            wav[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )
        self.val_wav_example["pred"] = (
            wav_pred[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )

    def on_validation_epoch_end(self) -> None:
        scheduler_list = self.lr_schedulers()
        scheduler_g = scheduler_list[0]
        scheduler_d = scheduler_list[1]
        scheduler_g.step()
        scheduler_d.step()

        self.train_epoch_loss_disc_f_list.append(
            np.mean(np.array(self.train_step_loss_disc_f_list))
        )
        self.train_epoch_loss_disc_s_list.append(
            np.mean(np.array(self.train_step_loss_disc_s_list))
        )
        self.train_epoch_loss_disc_all_list.append(
            np.mean(np.array(self.train_step_loss_disc_all_list))
        )
        self.train_epoch_loss_mel_list.append(
            np.mean(np.array(self.train_step_loss_mel_list))
        )
        self.train_epoch_loss_fm_f_list.append(
            np.mean(np.array(self.train_step_loss_fm_f_list))
        )
        self.train_epoch_loss_fm_s_list.append(
            np.mean(np.array(self.train_step_loss_fm_s_list))
        )
        self.train_epoch_loss_gen_f_list.append(
            np.mean(np.array(self.train_step_loss_gen_f_list))
        )
        self.train_epoch_loss_gen_s_list.append(
            np.mean(np.array(self.train_step_loss_gen_s_list))
        )
        self.train_epoch_loss_gen_all_list.append(
            np.mean(np.array(self.train_step_loss_gen_all_list))
        )
        self.val_epoch_loss_disc_f_list.append(
            np.mean(np.array(self.val_step_loss_disc_f_list))
        )
        self.val_epoch_loss_disc_s_list.append(
            np.mean(np.array(self.val_step_loss_disc_s_list))
        )
        self.val_epoch_loss_disc_all_list.append(
            np.mean(np.array(self.val_step_loss_disc_all_list))
        )
        self.val_epoch_loss_mel_list.append(
            np.mean(np.array(self.val_step_loss_mel_list))
        )
        self.val_epoch_loss_fm_f_list.append(
            np.mean(np.array(self.val_step_loss_fm_f_list))
        )
        self.val_epoch_loss_fm_s_list.append(
            np.mean(np.array(self.val_step_loss_fm_s_list))
        )
        self.val_epoch_loss_gen_f_list.append(
            np.mean(np.array(self.val_step_loss_gen_f_list))
        )
        self.val_epoch_loss_gen_s_list.append(
            np.mean(np.array(self.val_step_loss_gen_s_list))
        )
        self.val_epoch_loss_gen_all_list.append(
            np.mean(np.array(self.val_step_loss_gen_all_list))
        )
        self.train_step_loss_disc_f_list.clear()
        self.train_step_loss_disc_s_list.clear()
        self.train_step_loss_disc_all_list.clear()
        self.train_step_loss_mel_list.clear()
        self.train_step_loss_fm_f_list.clear()
        self.train_step_loss_fm_s_list.clear()
        self.train_step_loss_gen_f_list.clear()
        self.train_step_loss_gen_s_list.clear()
        self.train_step_loss_gen_all_list.clear()
        self.val_step_loss_disc_f_list.clear()
        self.val_step_loss_disc_s_list.clear()
        self.val_step_loss_disc_all_list.clear()
        self.val_step_loss_mel_list.clear()
        self.val_step_loss_fm_f_list.clear()
        self.val_step_loss_fm_s_list.clear()
        self.val_step_loss_gen_f_list.clear()
        self.val_step_loss_gen_s_list.clear()
        self.val_step_loss_gen_all_list.clear()

        save_epoch_loss_plot(
            title="loss_disc_f",
            train_loss_list=self.train_epoch_loss_disc_f_list,
            val_loss_list=self.val_epoch_loss_disc_f_list,
        )
        save_epoch_loss_plot(
            title="loss_disc_s",
            train_loss_list=self.train_epoch_loss_disc_s_list,
            val_loss_list=self.val_epoch_loss_disc_s_list,
        )
        save_epoch_loss_plot(
            title="loss_disc_all",
            train_loss_list=self.train_epoch_loss_disc_all_list,
            val_loss_list=self.val_epoch_loss_disc_all_list,
        )
        save_epoch_loss_plot(
            title="loss_mel",
            train_loss_list=self.train_epoch_loss_mel_list,
            val_loss_list=self.val_epoch_loss_mel_list,
        )
        save_epoch_loss_plot(
            title="loss_fm_f",
            train_loss_list=self.train_epoch_loss_fm_f_list,
            val_loss_list=self.val_epoch_loss_fm_f_list,
        )
        save_epoch_loss_plot(
            title="loss_fm_s",
            train_loss_list=self.train_epoch_loss_fm_s_list,
            val_loss_list=self.val_epoch_loss_fm_s_list,
        )
        save_epoch_loss_plot(
            title="loss_gen_f",
            train_loss_list=self.train_epoch_loss_gen_f_list,
            val_loss_list=self.val_epoch_loss_gen_f_list,
        )
        save_epoch_loss_plot(
            title="loss_gen_s",
            train_loss_list=self.train_epoch_loss_gen_s_list,
            val_loss_list=self.val_epoch_loss_gen_s_list,
        )
        save_epoch_loss_plot(
            title="loss_gen_all",
            train_loss_list=self.train_epoch_loss_gen_all_list,
            val_loss_list=self.val_epoch_loss_gen_all_list,
        )

        save_wav_table(
            cfg=self.cfg,
            gt_train=self.train_wav_example["gt"],
            pred_train=self.train_wav_example["pred"],
            gt_val=self.val_wav_example["gt"],
            pred_val=self.val_wav_example["pred"],
            tablename="wav examples",
        )

        save_mel(
            cfg=self.cfg,
            gt=wav2mel(wav=self.train_wav_example["gt"], cfg=self.cfg, ref_max=True),
            pred=wav2mel(
                wav=self.train_wav_example["pred"], cfg=self.cfg, ref_max=True
            ),
            filename="train",
        )
        save_mel(
            cfg=self.cfg,
            gt=wav2mel(wav=self.train_wav_example["gt"], cfg=self.cfg, ref_max=True),
            pred=wav2mel(
                wav=self.train_wav_example["pred"], cfg=self.cfg, ref_max=True
            ),
            filename="val",
        )

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            params=self.gen.parameters(),
            lr=self.learning_rate,
            betas=(
                self.cfg.training.optimizer.beta_1,
                self.cfg.training.optimizer.beta_2,
            ),
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )
        optimizer_d = torch.optim.AdamW(
            params=itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=self.learning_rate,
            betas=(
                self.cfg.training.optimizer.beta_1,
                self.cfg.training.optimizer.beta_2,
            ),
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g,
            gamma=self.cfg.training.scheduler.gamma,
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d,
            gamma=self.cfg.training.scheduler.gamma,
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


class LitHiFiGANMultipltSSLModelFineTuning(LitHiFiGANMultipltSSLModel):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__(cfg)
        self.lip2sp_module = LitWithSpeechSSLModule.load_from_checkpoint(
            checkpoint_path=cfg.training.lip2sp_model_path, cfg=cfg
        )

    def prepare_inputs_dict_using_prediction(
        self,
        pred_mel: torch.Tensor,
        pred_ssl_feature_cluster_linear: torch.Tensor,
        pred_ssl_feature_cluster_ssl: torch.Tensor,
        pred_ssl_feature_cluster_ensemble: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.cfg.model.decoder.vocoder_input_cluster == "avhubert":
            logger.debug("use avhubert prediction for hifigan inputs_dict.")
            inputs_dict = self.prepare_inputs_dict(
                mel=pred_mel,
                hubert_final_feature_cluster=pred_ssl_feature_cluster_linear.argmax(
                    dim=1
                ),
                wav2vec2_final_feature_cluster=None,
                data2vec_final_feature_cluster=None,
            )
        elif self.cfg.model.decoder.vocoder_input_cluster == "speech_ssl":
            logger.debug("use speech_ssl prediction for hifigan inputs_dict.")
            if (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-hubert-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=pred_ssl_feature_cluster_ssl.argmax(
                        dim=1
                    ),
                    wav2vec2_final_feature_cluster=None,
                    data2vec_final_feature_cluster=None,
                )
            elif (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-wav2vec2-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=None,
                    wav2vec2_final_feature_cluster=pred_ssl_feature_cluster_ssl.argmax(
                        dim=1
                    ),
                    data2vec_final_feature_cluster=None,
                )
            elif (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-data2vec-audio-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=None,
                    wav2vec2_final_feature_cluster=None,
                    data2vec_final_feature_cluster=pred_ssl_feature_cluster_ssl.argmax(
                        dim=1
                    ),
                )
        elif self.cfg.model.decoder.vocoder_input_cluster == "ensemble":
            logger.debug("use ensemble prediction for hifigan inputs_dict.")
            if (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-hubert-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=pred_ssl_feature_cluster_ensemble.argmax(
                        dim=1
                    ),
                    wav2vec2_final_feature_cluster=None,
                    data2vec_final_feature_cluster=None,
                )
            elif (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-wav2vec2-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=None,
                    wav2vec2_final_feature_cluster=pred_ssl_feature_cluster_ensemble.argmax(
                        dim=1
                    ),
                    data2vec_final_feature_cluster=None,
                )
            elif (
                self.cfg.model.decoder.speech_ssl.model_name
                == "rinna/japanese-data2vec-audio-base"
            ):
                inputs_dict = self.prepare_inputs_dict(
                    mel=pred_mel,
                    hubert_final_feature_cluster=None,
                    wav2vec2_final_feature_cluster=None,
                    data2vec_final_feature_cluster=pred_ssl_feature_cluster_ensemble.argmax(
                        dim=1
                    ),
                )
        return inputs_dict

    def calc_losses(self, batch: list) -> tuple:
        (
            wav,
            lip,
            feature,
            hubert_conv_feature,
            hubert_final_feature,
            hubert_final_feature_cluster,
            wav2vec2_conv_feature,
            wav2vec2_final_feature,
            wav2vec2_final_feature_cluster,
            data2vec_conv_feature,
            data2vec_final_feature,
            data2vec_final_feature_cluster,
            spk_emb,
            feature_len,
            feature_ssl_len,
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
        padding_mask_lip_inverse = ~padding_mask_lip
        padding_mask_feature = padding_mask_lip.repeat_interleave(
            repeats=get_upsample(self.cfg), dim=1
        )
        padding_mask_feature_speech_ssl = padding_mask_lip.repeat_interleave(
            repeats=get_upsample_speech_ssl(self.cfg), dim=1
        )

        with torch.no_grad():
            (
                pred_mel,
                pred_ssl_conv_feature,
                pred_ssl_feature_cluster_linear,
                pred_ssl_feature_cluster_ssl,
                pred_ssl_feature_cluster_ensemble,
            ) = self.lip2sp_module.model(
                lip=lip,
                audio=None,
                spk_emb=spk_emb,
                padding_mask_lip=padding_mask_lip,
                padding_mask_feature=padding_mask_lip_inverse,
            )

        pred_mel_lst = []
        pred_ssl_feature_cluster_linear_lst = []
        pred_ssl_feature_cluster_ssl_lst = []
        pred_ssl_feature_cluster_ensemble_lst = []
        wav_lst = []
        for i in range(self.cfg.training.batch_size):
            input_len_ssl_feature_cluster = int(
                self.cfg.training.input_sec_hifigan
                * self.cfg.data.video.fps
                * get_upsample_speech_ssl(self.cfg)
            )
            start_index_ssl_feature_cluster = random.randint(
                0,
                int(
                    min(
                        lip_len[i],
                        self.cfg.training.input_sec * self.cfg.data.video.fps,
                    )
                    * get_upsample_speech_ssl(self.cfg)
                )
                - input_len_ssl_feature_cluster,
            )
            input_len_mel = int(
                input_len_ssl_feature_cluster
                / get_upsample_speech_ssl(self.cfg)
                * get_upsample(self.cfg)
            )
            start_index_mel = int(
                start_index_ssl_feature_cluster
                / get_upsample_speech_ssl(self.cfg)
                * get_upsample(self.cfg)
            )
            input_len_wav = int(input_len_mel * self.cfg.data.audio.hop_length)
            start_index_wav = int(start_index_mel * self.cfg.data.audio.hop_length)

            pred_mel_lst.append(
                pred_mel[i, :, start_index_mel : start_index_mel + input_len_mel]
            )
            pred_ssl_feature_cluster_linear_lst.append(
                pred_ssl_feature_cluster_linear[
                    i,
                    :,
                    start_index_ssl_feature_cluster : start_index_ssl_feature_cluster
                    + input_len_ssl_feature_cluster,
                ]
            )
            pred_ssl_feature_cluster_ssl_lst.append(
                pred_ssl_feature_cluster_ssl[
                    i,
                    :,
                    start_index_ssl_feature_cluster : start_index_ssl_feature_cluster
                    + input_len_ssl_feature_cluster,
                ]
            )
            pred_ssl_feature_cluster_ensemble_lst.append(
                pred_ssl_feature_cluster_ensemble[
                    i,
                    :,
                    start_index_ssl_feature_cluster : start_index_ssl_feature_cluster
                    + input_len_ssl_feature_cluster,
                ]
            )
            wav_lst.append(wav[i, start_index_wav : start_index_wav + input_len_wav])

        pred_mel = torch.stack(pred_mel_lst, dim=0)
        pred_ssl_feature_cluster_linear = torch.stack(
            pred_ssl_feature_cluster_linear_lst, dim=0
        )
        pred_ssl_feature_cluster_ssl = torch.stack(
            pred_ssl_feature_cluster_ssl_lst, dim=0
        )
        pred_ssl_feature_cluster_ensemble = torch.stack(
            pred_ssl_feature_cluster_ssl_lst, dim=0
        )
        wav = torch.stack(wav_lst, dim=0)

        inputs_dict = self.prepare_inputs_dict_using_prediction(
            pred_mel=pred_mel,
            pred_ssl_feature_cluster_linear=pred_ssl_feature_cluster_linear,
            pred_ssl_feature_cluster_ssl=pred_ssl_feature_cluster_ssl,
            pred_ssl_feature_cluster_ensemble=pred_ssl_feature_cluster_ensemble,
        )

        wav = wav.unsqueeze(1)
        mel = self.wav2mel(wav)

        wav_pred = self.gen(inputs_dict, spk_emb)
        mel_pred = self.wav2mel(wav_pred)

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(wav, wav_pred.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(wav, wav_pred.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )
        loss_disc_all = loss_disc_s + loss_disc_f

        loss_mel = torch.nn.functional.l1_loss(mel, mel_pred) * 45
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(wav, wav_pred)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(wav, wav_pred)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        return (
            wav,
            wav_pred,
            loss_disc_f,
            loss_disc_s,
            loss_disc_all,
            loss_mel,
            loss_fm_f,
            loss_fm_s,
            loss_gen_f,
            loss_gen_s,
            loss_gen_all,
        )

    def test_step(self, batch: list, batch_index: int) -> None:
        (
            wav_gt,
            lip,
            feature,
            hubert_conv_feature,
            hubert_final_feature,
            hubert_final_feature_cluster,
            wav2vec2_conv_feature,
            wav2vec2_final_feature,
            wav2vec2_final_feature_cluster,
            data2vec_conv_feature,
            data2vec_final_feature,
            data2vec_final_feature_cluster,
            spk_emb,
            feature_len,
            feature_ssl_len,
            lip_len,
            speaker,
            filename,
        ) = batch

        lip = lip.to(torch.float32)

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
        padding_mask_feature_speech_ssl = padding_mask_lip.repeat_interleave(
            repeats=get_upsample_speech_ssl(self.cfg), dim=1
        )

        (
            pred_mel,
            pred_ssl_conv_feature,
            pred_ssl_feature_cluster_linear,
            pred_ssl_feature_cluster_ssl,
            pred_ssl_feature_cluster_ensemble,
        ) = self.lip2sp_module.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            padding_mask_lip=padding_mask_lip,
            padding_mask_feature=padding_mask_lip,
        )

        inputs_dict = self.prepare_inputs_dict_using_prediction(
            pred_mel=pred_mel,
            pred_ssl_feature_cluster_linear=pred_ssl_feature_cluster_linear,
            pred_ssl_feature_cluster_ssl=pred_ssl_feature_cluster_ssl,
            pred_ssl_feature_cluster_ensemble=pred_ssl_feature_cluster_ensemble,
        )

        wav_pred = self.gen(inputs_dict, spk_emb)

        inputs_dict = self.prepare_inputs_dict(
            mel=feature,
            hubert_final_feature_cluster=hubert_final_feature_cluster,
            wav2vec2_final_feature_cluster=wav2vec2_final_feature_cluster,
            data2vec_final_feature_cluster=data2vec_final_feature_cluster,
        )
        wav_abs = self.gen(inputs_dict, spk_emb)

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

        save_dir = (
            Path(
                str(Path(self.cfg.training.checkpoints_save_dir)).replace(
                    "checkpoints", "results"
                )
            )
            / speaker[0]
            / filename[0]
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        write(
            filename=str(save_dir / "gt.wav"),
            rate=self.cfg.data.audio.sr,
            data=wav_gt.cpu().numpy(),
        )
        write(
            filename=str(save_dir / "abs.wav"),
            rate=self.cfg.data.audio.sr,
            data=wav_abs.cpu().numpy(),
        )
        write(
            filename=str(save_dir / "pred.wav"),
            rate=self.cfg.data.audio.sr,
            data=wav_pred.cpu().numpy(),
        )

    def on_test_start(self) -> None:
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
        return wer_gt
