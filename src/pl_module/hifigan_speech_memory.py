import itertools
import logging

import librosa
import lightning as L
import numpy as np
import omegaconf
import torch

from src.data_process.utils import wav2mel
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel, save_wav_table
from src.model.hifigan_speech_memory import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LitHiFiGANSpeechMemoryModel(L.LightningModule):
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
        hubert_layer_feature_cluster: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        """
        args:
            mel: (B, C, T)
            hubert_layer_feature_cluster: (B, T)
        """
        if mel is not None:
            mel = mel.permute(0, 2, 1)
            mel = mel.reshape(mel.shape[0], -1, int(self.cfg.data.audio.n_mels * 2))

        inputs_dict = {
            "mel": mel,
            "hubert_layer_feature_cluster": hubert_layer_feature_cluster,
        }
        return inputs_dict

    def calc_losses(self, batch: list) -> tuple:
        (
            wav,
            lip,
            feature,
            hubert_layer_feature_cluster,
            spk_emb,
            feature_len,
            feature_ssl_len,
            lip_len,
            speaker,
            filename,
        ) = batch
        wav = wav.unsqueeze(1)
        mel = self.wav2mel(wav)

        inputs_dict = self.prepare_inputs_dict(feature, hubert_layer_feature_cluster)

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
