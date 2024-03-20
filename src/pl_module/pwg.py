from pathlib import Path

import lightning as L
import MeCab
import numpy as np
import omegaconf
import pandas as pd
import torch
import wandb
import whisper
from jiwer import wer
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.data_process.utils import wav2mel
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel, save_wav_table
from src.loss_fn.stft_loss import MultiResolutionSTFTLoss
from src.model.pwg import Discriminator, Generator


class LitPWG(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg["training"]["optimizer"]["learning_rate"]
        self.automatic_optimization = False
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)
        self.loss_fn = MultiResolutionSTFTLoss(
            n_fft_list=cfg["training"]["params"]["n_fft_list"],
            hop_length_list=cfg["training"]["params"]["hop_length_list"],
            win_length_list=cfg["training"]["params"]["win_length_list"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.train_step_loss_disc_list = []
        self.train_step_loss_gen_stft_list = []
        self.train_step_loss_gen_gan_list = []
        self.train_step_loss_gen_all_list = []
        self.train_epoch_loss_disc_list = []
        self.train_epoch_loss_gen_stft_list = []
        self.train_epoch_loss_gen_gan_list = []
        self.train_epoch_loss_gen_all_list = []
        self.val_step_loss_disc_list = []
        self.val_step_loss_gen_stft_list = []
        self.val_step_loss_gen_gan_list = []
        self.val_step_loss_gen_all_list = []
        self.val_epoch_loss_disc_list = []
        self.val_epoch_loss_gen_stft_list = []
        self.val_epoch_loss_gen_gan_list = []
        self.val_epoch_loss_gen_all_list = []
        self.train_wav_example = {"gt": None, "pred": None}
        self.val_wav_example = {"gt": None, "pred": None}

    def forward(self, noise: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        wav_pred = self.generator(noise, feature)
        return wav_pred

    def training_step(self, batch: list, batch_index: int) -> None:
        opt_g, opt_d = self.optimizers()

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
        wav = wav.unsqueeze(1)

        noise = torch.randn(
            feature.shape[0],
            1,
            feature.shape[-1] * self.cfg["data"]["audio"]["hop_length"],
        ).to(device=self.device, dtype=feature.dtype)
        wav_pred = self.generator(noise, feature)

        self.toggle_optimizer(opt_d)
        out_real = self.discriminator(wav)
        out_pred = self.discriminator(wav_pred.detach())
        loss_disc = torch.mean((out_real - 1) ** 2) + torch.mean(out_pred**2)
        self.manual_backward(loss_disc)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.toggle_optimizer(opt_g)
        out_pred = self.discriminator(wav_pred)
        loss_gen_stft = self.loss_fn.calc_loss(wav, wav_pred)
        loss_gen_gan = torch.mean((out_pred - 1) ** 2)
        loss_gen_all = (
            self.cfg["training"]["params"]["stft_loss_weight"] * loss_gen_stft
            + self.cfg["training"]["params"]["gan_loss_weight"] * loss_gen_gan
        )
        self.manual_backward(loss_gen_all)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log(
            "train_loss_disc",
            loss_disc,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "train_loss_gen_stft",
            loss_gen_stft,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "train_loss_gen_gan",
            loss_gen_gan,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "train_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.train_step_loss_disc_list.append(loss_disc.item())
        self.train_step_loss_gen_stft_list.append(loss_gen_stft.item())
        self.train_step_loss_gen_gan_list.append(loss_gen_gan.item())
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
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            filename,
        ) = batch
        wav = wav.unsqueeze(1)

        noise = torch.randn(
            feature.shape[0],
            1,
            feature.shape[-1] * self.cfg["data"]["audio"]["hop_length"],
        ).to(device=self.device, dtype=feature.dtype)
        wav_pred = self.generator(noise, feature)

        out_real = self.discriminator(wav)
        out_pred = self.discriminator(wav_pred)

        loss_disc = torch.mean((out_real - 1) ** 2) + torch.mean(out_pred**2)
        loss_gen_stft = self.loss_fn.calc_loss(wav, wav_pred)
        loss_gen_gan = torch.mean((out_pred - 1) ** 2)
        loss_gen_all = (
            self.cfg["training"]["params"]["stft_loss_weight"] * loss_gen_stft
            + self.cfg["training"]["params"]["gan_loss_weight"] * loss_gen_gan
        )

        self.log(
            "val_loss_disc",
            loss_disc,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "val_loss_gen_stft",
            loss_gen_stft,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "val_loss_gen_gan",
            loss_gen_gan,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.log(
            "val_loss_gen_all",
            loss_gen_all,
            logger=True,
            batch_size=self.cfg["training"]["params"]["batch_size"],
        )
        self.val_step_loss_disc_list.append(loss_disc.item())
        self.val_step_loss_gen_stft_list.append(loss_gen_stft.item())
        self.val_step_loss_gen_gan_list.append(loss_gen_gan.item())
        self.val_step_loss_gen_all_list.append(loss_gen_all.item())
        self.val_wav_example["gt"] = (
            wav[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )
        self.val_wav_example["pred"] = (
            wav_pred[0].cpu().detach().numpy().astype(np.float32).squeeze(0)
        )

    def on_validation_epoch_end(self) -> None:
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        
        self.train_epoch_loss_disc_list.append(np.mean(self.train_step_loss_disc_list))
        self.train_epoch_loss_gen_stft_list.append(
            np.mean(self.train_step_loss_gen_stft_list)
        )
        self.train_epoch_loss_gen_gan_list.append(
            np.mean(self.train_step_loss_gen_gan_list)
        )
        self.train_epoch_loss_gen_all_list.append(
            np.mean(self.train_step_loss_gen_all_list)
        )
        self.train_step_loss_disc_list.clear()
        self.train_step_loss_gen_stft_list.clear()
        self.train_step_loss_gen_gan_list.clear()
        self.train_step_loss_gen_all_list.clear()

        self.val_epoch_loss_disc_list.append(np.mean(self.val_step_loss_disc_list))
        self.val_epoch_loss_gen_stft_list.append(
            np.mean(self.val_step_loss_gen_stft_list)
        )
        self.val_epoch_loss_gen_gan_list.append(
            np.mean(self.val_step_loss_gen_gan_list)
        )
        self.val_epoch_loss_gen_all_list.append(
            np.mean(self.val_step_loss_gen_all_list)
        )
        self.val_step_loss_disc_list.clear()
        self.val_step_loss_gen_stft_list.clear()
        self.val_step_loss_gen_gan_list.clear()
        self.val_step_loss_gen_all_list.clear()

        save_epoch_loss_plot(
            title="loss_disc",
            train_loss_list=self.train_epoch_loss_disc_list,
            val_loss_list=self.val_epoch_loss_disc_list,
        )
        save_epoch_loss_plot(
            title="loss_gen_stft",
            train_loss_list=self.train_epoch_loss_gen_stft_list,
            val_loss_list=self.val_epoch_loss_gen_stft_list,
        )
        save_epoch_loss_plot(
            title="loss_gen_gan",
            train_loss_list=self.train_epoch_loss_gen_gan_list,
            val_loss_list=self.val_epoch_loss_gen_gan_list,
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

    def test_step(self, batch: list, batch_index: int) -> None:
        (
            wav_gt,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            filename,
        ) = batch

        noise = torch.randn(
            feature.shape[0],
            1,
            feature.shape[-1] * self.cfg["data"]["audio"]["hop_length"],
        ).to(device=self.device, dtype=feature.dtype)
        wav_pred = self.generator(noise, feature)

        n_sample_min = min(wav_gt.shape[-1], wav_pred.shape[-1])
        wav_gt = self.process_wav(wav_gt, n_sample_min)
        wav_pred = self.process_wav(wav_pred, n_sample_min)

        pesq_pred = self.wb_pesq_evaluator(wav_pred, wav_gt)
        stoi_pred = self.stoi_evaluator(wav_pred, wav_gt)
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
        utt_recog_pred = (
            self.speech_recognizer.transcribe(
                wav_pred.to(dtype=torch.float32), language="ja"
            )["text"]
            .replace("。", "")
            .replace("、", "")
        )

        utt_parse = self.mecab.parse(utt)
        utt_recog_gt_parse = self.mecab.parse(utt_recog_gt)
        utt_recog_pred_parse = self.mecab.parse(utt_recog_pred)

        wer_gt = self.calc_error_rate(utt_parse, utt_recog_gt_parse)
        wer_pred = self.calc_error_rate(utt_parse, utt_recog_pred_parse)

        data = [
            [
                speaker[0],
                filename[0],
                "gt",
                wandb.Audio(wav_gt.cpu(), sample_rate=self.cfg["data"]["audio"]["sr"]),
                None,
                None,
                None,
                wer_gt,
            ],
            [
                speaker[0],
                filename[0],
                "pred",
                wandb.Audio(
                    wav_pred.cpu(), sample_rate=self.cfg["data"]["audio"]["sr"]
                ),
                pesq_pred,
                stoi_pred,
                estoi_pred,
                wer_pred,
            ],
        ]
        self.test_data_list += data

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
            self.cfg["data"]["audio"]["sr"], "wb"
        )
        self.stoi_evaluator = ShortTimeObjectiveIntelligibility(
            self.cfg["data"]["audio"]["sr"], extended=False
        )
        self.estoi_evaluator = ShortTimeObjectiveIntelligibility(
            self.cfg["data"]["audio"]["sr"], extended=True
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
        optimizer_g = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.learning_rate,
            betas=(
                self.cfg["training"]["optimizer"]["beta_1"],
                self.cfg["training"]["optimizer"]["beta_2"],
            ),
            weight_decay=self.cfg["training"]["optimizer"]["weight_decay"],
        )
        optimizer_d = torch.optim.Adam(
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
