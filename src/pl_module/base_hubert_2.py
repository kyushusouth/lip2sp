import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jiwer
import lightning as L
import MeCab
import neologdn
from num2words import num2words
import numpy as np
import omegaconf
import pandas as pd
import polars as pl
import pyopenjtalk
import torch
import whisper
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from scipy.io.wavfile import write
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

import wandb
from src.data_process.utils import get_upsample, get_upsample_speech_ssl
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel
from src.loss_fn.base import LossFunctions
from src.model.base_hubert_2 import BaseHuBERT2Model
from src.pl_module.hifigan_base_hubert_2 import LitHiFiGANBaseHuBERT2Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TotalErrors:
    deletions: int = 0
    insertions: int = 0
    substitutions: int = 0
    distance: int = 0
    length: int = 0

    @property
    def error_rate(self):
        return self.distance / self.length


class LitBaseHuBERT2Module(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.automatic_optimization = True

        self.model = BaseHuBERT2Model(cfg)

        if "avhubert" in cfg.model.freeze:
            logger.info("freeze avhubert.")
            for param in self.model.avhubert.parameters():
                param.requires_grad = False
        if "decoders_avhubert" in cfg.model.freeze:
            logger.info("freeze decoders_avhubert.")
            for param in self.model.decoders_avhubert.parameters():
                param.requires_grad = False
        if "ssl_model_encoder" in cfg.model.freeze:
            logger.info("freeze ssl_model_encoder.")
            for param in self.model.ssl_model_encoder.parameters():
                param.requires_grad = False
        if "decoders_hubert" in cfg.model.freeze:
            logger.info("freeze decoders_hubert.")
            for param in self.model.decoders_hubert.parameters():
                param.requires_grad = False
        if "ensemble_encoder" in cfg.model.freeze:
            logger.info("freeze ensemble_encoder.")
            for param in self.model.ensemble_encoder.parameters():
                param.requires_grad = False
        if "decoders_ensemble" in cfg.model.freeze:
            logger.info("freeze decoders_ensemble")
            for param in self.model.decoders_ensemble.parameters():
                param.requires_grad = False

        if (
            "ssl_model_encoder" not in cfg.model.freeze
        ) and cfg.model.decoder.speech_ssl.partial_update.use:
            logger.info("Freeze Some Speech SSL Layers for Partial Update")
            for layer_index, layer in enumerate(
                self.model.ssl_model_encoder.ssl_model_encoder.layers
            ):
                if (
                    self.cfg.model.decoder.speech_ssl.partial_update.lower_or_upper
                    == "lower"
                    and layer_index
                    < self.cfg.model.decoder.speech_ssl.partial_update.thres
                ) or (
                    self.cfg.model.decoder.speech_ssl.partial_update.lower_or_upper
                    == "upper"
                    and layer_index
                    > self.cfg.model.decoder.speech_ssl.partial_update.thres
                ):
                    logger.info(f"freeze: {layer_index}")
                    for param in layer.parameters():
                        param.requires_grad = False

        self.loss_fn = LossFunctions()

        self.train_step_mel_loss_lst = []
        self.train_step_ssl_conv_feature_loss_lst = []
        self.train_step_ssl_feature_cluster_loss_lst = []
        self.train_step_mel_speech_ssl_loss_lst = []
        self.train_step_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.train_step_mel_ensemble_loss_lst = []
        self.train_step_ssl_feature_cluster_ensemble_loss_lst = []
        self.train_step_total_loss_lst = []
        self.train_epoch_mel_loss_lst = []
        self.train_epoch_ssl_conv_feature_loss_lst = []
        self.train_epoch_ssl_feature_cluster_loss_lst = []
        self.train_epoch_mel_speech_ssl_loss_lst = []
        self.train_epoch_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.train_epoch_mel_ensemble_loss_lst = []
        self.train_epoch_ssl_feature_cluster_ensemble_loss_lst = []
        self.train_epoch_total_loss_lst = []
        self.val_step_mel_loss_lst = []
        self.val_step_ssl_conv_feature_loss_lst = []
        self.val_step_ssl_feature_cluster_loss_lst = []
        self.val_step_mel_speech_ssl_loss_lst = []
        self.val_step_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.val_step_mel_ensemble_loss_lst = []
        self.val_step_ssl_feature_cluster_ensemble_loss_lst = []
        self.val_step_total_loss_lst = []
        self.val_epoch_mel_loss_lst = []
        self.val_epoch_ssl_conv_feature_loss_lst = []
        self.val_epoch_ssl_feature_cluster_loss_lst = []
        self.val_epoch_mel_speech_ssl_loss_lst = []
        self.val_epoch_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.val_epoch_mel_ensemble_loss_lst = []
        self.val_epoch_ssl_feature_cluster_ensemble_loss_lst = []
        self.val_epoch_total_loss_lst = []
        self.train_mel_example = {
            "gt": None,
            "pred_mel": None,
            "pred_mel_speech_ssl": None,
            "pred_mel_ensemble": None,
        }
        self.val_mel_example = {
            "gt": None,
            "pred_mel": None,
            "pred_mel_speech_ssl": None,
            "pred_mel_ensemble": None,
        }

    def calc_losses(self, batch: list) -> tuple:
        (
            wav,
            lip,
            feature,
            hubert_conv_feature,
            hubert_layer_feature_cluster,
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
        padding_mask_feature = padding_mask_lip.repeat_interleave(
            repeats=get_upsample(self.cfg), dim=1
        )
        padding_mask_feature_speech_ssl = padding_mask_lip.repeat_interleave(
            repeats=get_upsample_speech_ssl(self.cfg), dim=1
        )

        (
            pred_mel,
            pred_ssl_conv_feature,
            pred_ssl_feature_cluster,
            pred_mel_speech_ssl,
            pred_ssl_feature_cluster_speech_ssl,
            pred_mel_ensemble,
            pred_ssl_feature_cluster_ensemble,
        ) = self.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            padding_mask_lip=padding_mask_lip,
            padding_mask_speech_ssl=~padding_mask_feature_speech_ssl,
        )

        mel_loss = self.loss_fn.l1_loss(
            pred=pred_mel,
            target=feature,
            mask=padding_mask_feature,
        )
        ssl_conv_feature_loss = self.loss_fn.l1_loss(
            pred=pred_ssl_conv_feature,
            target=hubert_conv_feature,
            mask=padding_mask_feature_speech_ssl,
        )
        ssl_feature_cluster_loss = torch.nn.functional.cross_entropy(
            input=pred_ssl_feature_cluster,
            target=hubert_layer_feature_cluster,
            ignore_index=0,
        )
        mel_speech_ssl_loss = self.loss_fn.l1_loss(
            pred=pred_mel_speech_ssl,
            target=feature,
            mask=padding_mask_feature,
        )
        ssl_feature_cluster_speech_ssl_loss = torch.nn.functional.cross_entropy(
            input=pred_ssl_feature_cluster_speech_ssl,
            target=hubert_layer_feature_cluster,
            ignore_index=0,
        )
        mel_ensemble_loss = self.loss_fn.l1_loss(
            pred=pred_mel_ensemble,
            target=feature,
            mask=padding_mask_feature,
        )
        ssl_feature_cluster_ensemble_loss = torch.nn.functional.cross_entropy(
            input=pred_ssl_feature_cluster_ensemble,
            target=hubert_layer_feature_cluster,
            ignore_index=0,
        )

        total_loss = (
            (mel_loss * self.cfg.training.loss_weights.mel_loss)
            + (
                ssl_conv_feature_loss
                * self.cfg.training.loss_weights.ssl_conv_feature_loss
            )
            + (
                ssl_feature_cluster_loss
                * self.cfg.training.loss_weights.ssl_feature_cluster_loss
            )
            + (mel_speech_ssl_loss * self.cfg.training.loss_weights.mel_speech_ssl_loss)
            + (
                ssl_feature_cluster_speech_ssl_loss
                * self.cfg.training.loss_weights.ssl_feature_cluster_speech_ssl_loss
            )
            + (mel_ensemble_loss * self.cfg.training.loss_weights.mel_ensemble_loss)
            + (
                ssl_feature_cluster_ensemble_loss
                * self.cfg.training.loss_weights.ssl_feature_cluster_ensemble_loss
            )
        )

        return (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_loss,
            mel_speech_ssl_loss,
            ssl_feature_cluster_speech_ssl_loss,
            mel_ensemble_loss,
            ssl_feature_cluster_ensemble_loss,
            total_loss,
            feature,
            pred_mel,
            pred_mel_speech_ssl,
            pred_mel_ensemble,
        )

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_loss,
            mel_speech_ssl_loss,
            ssl_feature_cluster_speech_ssl_loss,
            mel_ensemble_loss,
            ssl_feature_cluster_ensemble_loss,
            total_loss,
            feature,
            pred_mel,
            pred_mel_speech_ssl,
            pred_mel_ensemble,
        ) = self.calc_losses(batch)

        self.log(
            "train_mel_loss",
            mel_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_ssl_conv_feature_loss",
            ssl_conv_feature_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_ssl_feature_cluster_loss",
            ssl_feature_cluster_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_mel_speech_ssl_loss",
            mel_speech_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_ssl_feature_cluster_speech_ssl_loss",
            ssl_feature_cluster_speech_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_mel_ensemble_loss",
            mel_ensemble_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_ssl_feature_cluster_ensemble_loss",
            ssl_feature_cluster_ensemble_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.train_step_mel_loss_lst.append(mel_loss.item())
        self.train_step_ssl_conv_feature_loss_lst.append(ssl_conv_feature_loss.item())
        self.train_step_ssl_feature_cluster_loss_lst.append(
            ssl_feature_cluster_loss.item()
        )
        self.train_step_mel_speech_ssl_loss_lst.append(mel_speech_ssl_loss.item())
        self.train_step_ssl_feature_cluster_speech_ssl_loss_lst.append(
            ssl_feature_cluster_speech_ssl_loss.item()
        )
        self.train_step_mel_ensemble_loss_lst.append(mel_ensemble_loss.item())
        self.train_step_ssl_feature_cluster_ensemble_loss_lst.append(
            ssl_feature_cluster_ensemble_loss.item()
        )
        self.train_step_total_loss_lst.append(total_loss.item())

        self.train_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.train_mel_example["pred_mel"] = (
            pred_mel[0].cpu().detach().numpy().astype(np.float32)
        )
        self.train_mel_example["pred_mel_speech_ssl"] = (
            pred_mel_speech_ssl[0].cpu().detach().numpy().astype(np.float32)
        )
        self.train_mel_example["pred_mel_ensemble"] = (
            pred_mel_ensemble[0].cpu().detach().numpy().astype(np.float32)
        )

        return total_loss

    def validation_step(self, batch: list, batch_index: int) -> None:
        (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_loss,
            mel_speech_ssl_loss,
            ssl_feature_cluster_speech_ssl_loss,
            mel_ensemble_loss,
            ssl_feature_cluster_ensemble_loss,
            total_loss,
            feature,
            pred_mel,
            pred_mel_speech_ssl,
            pred_mel_ensemble,
        ) = self.calc_losses(batch)

        self.log(
            "val_mel_loss",
            mel_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_ssl_conv_feature_loss",
            ssl_conv_feature_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_ssl_feature_cluster_loss",
            ssl_feature_cluster_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_mel_speech_ssl_loss",
            mel_speech_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_ssl_feature_cluster_speech_ssl_loss",
            ssl_feature_cluster_speech_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_mel_ensemble_loss",
            mel_ensemble_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_ssl_feature_cluster_ensemble_loss",
            ssl_feature_cluster_ensemble_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.val_step_mel_loss_lst.append(mel_loss.item())
        self.val_step_ssl_conv_feature_loss_lst.append(ssl_conv_feature_loss.item())
        self.val_step_ssl_feature_cluster_loss_lst.append(
            ssl_feature_cluster_loss.item()
        )
        self.val_step_mel_speech_ssl_loss_lst.append(mel_speech_ssl_loss.item())
        self.val_step_ssl_feature_cluster_speech_ssl_loss_lst.append(
            ssl_feature_cluster_speech_ssl_loss.item()
        )
        self.val_step_mel_ensemble_loss_lst.append(mel_ensemble_loss.item())
        self.val_step_ssl_feature_cluster_ensemble_loss_lst.append(
            ssl_feature_cluster_ensemble_loss.item()
        )
        self.val_step_total_loss_lst.append(total_loss.item())

        self.val_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.val_mel_example["pred_mel"] = (
            pred_mel[0].cpu().detach().numpy().astype(np.float32)
        )
        self.val_mel_example["pred_mel_speech_ssl"] = (
            pred_mel_speech_ssl[0].cpu().detach().numpy().astype(np.float32)
        )
        self.val_mel_example["pred_mel_ensemble"] = (
            pred_mel_ensemble[0].cpu().detach().numpy().astype(np.float32)
        )

    def on_validation_epoch_end(self) -> None:
        self.train_epoch_mel_loss_lst.append(np.mean(self.train_step_mel_loss_lst))
        self.train_epoch_ssl_conv_feature_loss_lst.append(
            np.mean(self.train_step_ssl_conv_feature_loss_lst)
        )
        self.train_epoch_ssl_feature_cluster_loss_lst.append(
            np.mean(self.train_step_ssl_feature_cluster_loss_lst)
        )
        self.train_epoch_mel_speech_ssl_loss_lst.append(
            np.mean(self.train_step_mel_speech_ssl_loss_lst)
        )
        self.train_epoch_ssl_feature_cluster_speech_ssl_loss_lst.append(
            np.mean(self.train_step_ssl_feature_cluster_speech_ssl_loss_lst)
        )
        self.train_epoch_mel_ensemble_loss_lst.append(
            np.mean(self.train_step_mel_ensemble_loss_lst)
        )
        self.train_epoch_ssl_feature_cluster_ensemble_loss_lst.append(
            np.mean(self.train_step_ssl_feature_cluster_ensemble_loss_lst)
        )
        self.train_epoch_total_loss_lst.append(np.mean(self.train_step_total_loss_lst))
        self.train_step_mel_loss_lst = []
        self.train_step_ssl_conv_feature_loss_lst = []
        self.train_step_ssl_feature_cluster_loss_lst = []
        self.train_step_mel_speech_ssl_loss_lst = []
        self.train_step_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.train_step_mel_ensemble_loss_lst = []
        self.train_step_ssl_feature_cluster_ensemble_loss_lst = []
        self.train_step_total_loss_lst = []

        self.val_epoch_mel_loss_lst.append(np.mean(self.val_step_mel_loss_lst))
        self.val_epoch_ssl_conv_feature_loss_lst.append(
            np.mean(self.val_step_ssl_conv_feature_loss_lst)
        )
        self.val_epoch_ssl_feature_cluster_loss_lst.append(
            np.mean(self.val_step_ssl_feature_cluster_loss_lst)
        )
        self.val_epoch_mel_speech_ssl_loss_lst.append(
            np.mean(self.val_step_mel_speech_ssl_loss_lst)
        )
        self.val_epoch_ssl_feature_cluster_speech_ssl_loss_lst.append(
            np.mean(self.val_step_ssl_feature_cluster_speech_ssl_loss_lst)
        )
        self.val_epoch_mel_ensemble_loss_lst.append(
            np.mean(self.val_step_mel_ensemble_loss_lst)
        )
        self.val_epoch_ssl_feature_cluster_ensemble_loss_lst.append(
            np.mean(self.val_step_ssl_feature_cluster_ensemble_loss_lst)
        )
        self.val_epoch_total_loss_lst.append(np.mean(self.val_step_total_loss_lst))
        self.val_step_mel_loss_lst = []
        self.val_step_ssl_conv_feature_loss_lst = []
        self.val_step_ssl_feature_cluster_loss_lst = []
        self.val_step_mel_speech_ssl_loss_lst = []
        self.val_step_ssl_feature_cluster_speech_ssl_loss_lst = []
        self.val_step_mel_ensemble_loss_lst = []
        self.val_step_ssl_feature_cluster_ensemble_loss_lst = []
        self.val_step_total_loss_lst = []

        save_epoch_loss_plot(
            title="mel_loss",
            train_loss_list=self.train_epoch_mel_loss_lst,
            val_loss_list=self.val_epoch_mel_loss_lst,
        )
        save_epoch_loss_plot(
            title="ssl_conv_feature_loss",
            train_loss_list=self.train_epoch_ssl_conv_feature_loss_lst,
            val_loss_list=self.val_epoch_ssl_conv_feature_loss_lst,
        )
        save_epoch_loss_plot(
            title="ssl_feature_cluster_loss",
            train_loss_list=self.train_epoch_ssl_feature_cluster_loss_lst,
            val_loss_list=self.val_epoch_ssl_feature_cluster_loss_lst,
        )
        save_epoch_loss_plot(
            title="mel_speech_ssl_loss",
            train_loss_list=self.train_epoch_mel_speech_ssl_loss_lst,
            val_loss_list=self.val_epoch_mel_speech_ssl_loss_lst,
        )
        save_epoch_loss_plot(
            title="ssl_feature_cluster_speech_ssl_loss",
            train_loss_list=self.train_epoch_ssl_feature_cluster_speech_ssl_loss_lst,
            val_loss_list=self.val_epoch_ssl_feature_cluster_speech_ssl_loss_lst,
        )
        save_epoch_loss_plot(
            title="mel_ensemble_loss",
            train_loss_list=self.train_epoch_mel_ensemble_loss_lst,
            val_loss_list=self.val_epoch_mel_ensemble_loss_lst,
        )
        save_epoch_loss_plot(
            title="ssl_feature_cluster_ensemble_loss",
            train_loss_list=self.train_epoch_ssl_feature_cluster_ensemble_loss_lst,
            val_loss_list=self.val_epoch_ssl_feature_cluster_ensemble_loss_lst,
        )
        save_epoch_loss_plot(
            title="total_loss",
            train_loss_list=self.train_epoch_total_loss_lst,
            val_loss_list=self.val_epoch_total_loss_lst,
        )

        save_mel(
            cfg=self.cfg,
            gt=self.train_mel_example["gt"],
            pred=self.train_mel_example["pred_mel"],
            filename="train_pred_mel",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.train_mel_example["gt"],
            pred=self.train_mel_example["pred_mel_speech_ssl"],
            filename="train_pred_mel_speech_ssl",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.train_mel_example["gt"],
            pred=self.train_mel_example["pred_mel_ensemble"],
            filename="train_pred_mel_ensemble",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.val_mel_example["gt"],
            pred=self.val_mel_example["pred_mel"],
            filename="val_pred_mel",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.val_mel_example["gt"],
            pred=self.val_mel_example["pred_mel_speech_ssl"],
            filename="val_pred_mel_speech_ssl",
        )
        save_mel(
            cfg=self.cfg,
            gt=self.val_mel_example["gt"],
            pred=self.val_mel_example["pred_mel_ensemble"],
            filename="val_pred_mel_ensemble",
        )

    def on_test_start(self) -> None:
        self.model.eval()

        self.test_data_columns = [
            "speaker",
            "filename",
            "kind",
            "wav",
            "pesq",
            "stoi",
            "estoi",
            "utt_gt_kanjikana",
            "utt_recog_kanjikana",
            "delt_kanjikana",
            "inst_kanjikana",
            "subst_kanjikana",
            "wer_kanjikana",
            "utt_gt_kana",
            "utt_recog_kana",
            "delt_kana",
            "inst_kana",
            "subst_kana",
            "cer_kana",
            "utt_gt_phoneme",
            "utt_recog_phoneme",
            "delt_phoneme",
            "inst_phoneme",
            "subst_phoneme",
            "per_phoneme",
            "spk_sim",
            "utmos",
        ]
        self.test_data_list = []

        self.total_errors_dct = {
            "gt": {
                "kanjikana": TotalErrors(),
                "kana": TotalErrors(),
                "phoneme": TotalErrors(),
            },
            "abs_mel_speech_ssl": {
                "kanjikana": TotalErrors(),
                "kana": TotalErrors(),
                "phoneme": TotalErrors(),
            },
            "pred_mel_speech_ssl": {
                "kanjikana": TotalErrors(),
                "kana": TotalErrors(),
                "phoneme": TotalErrors(),
            },
        }

        csv_path = Path(self.cfg.path.kablab.utt_csv_test_path).expanduser()
        self.df_utt = pd.read_csv(str(csv_path))
        self.df_utt = self.df_utt.drop(columns=["index"])
        self.df_utt = self.df_utt.loc[self.df_utt["subset"] == "j"]

        self.hifigan_mel_speech_ssl = LitHiFiGANBaseHuBERT2Model.load_from_checkpoint(
            self.cfg.model.hifigan.model_path_mel_speech_ssl,
            cfg=self.cfg,
            input_type=["mel", "hubert_layer_feature_cluster"],
        )
        self.hifigan_mel_speech_ssl.eval()

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

    def process_wav(self, wav: torch.Tensor, n_sample: int) -> torch.Tensor:
        wav = wav.to(torch.float32)
        wav = wav.squeeze(0).squeeze(0)
        wav /= torch.max(torch.abs(wav))
        wav = wav[:n_sample]
        return wav

    def normalize_japanese_text(self, utt_txt: str) -> str:
        symbols = ["、", "。", " ", "!", "?", "・", "\n"]
        _SPECIALS = {ord(c): "" for c in symbols}
        utt_txt = utt_txt.translate(_SPECIALS)

        utt_txt = neologdn.normalize(utt_txt)

        numbers = re.findall(r"\d+\.?\d*", utt_txt)
        transcribed_numbers = [num2words(item, lang="ja") for item in numbers]
        for nr in range(len(numbers)):
            old_nr = numbers[nr]
            new_nr = transcribed_numbers[nr]
            utt_txt = utt_txt.replace(old_nr, new_nr, 1)
        return utt_txt

    def preprocess_text_for_error_rate(self, kanjikana: str) -> tuple[str, str, str]:
        """誤り率のためのテキスト前処理
        args:
            kanjikana: Whisperから得られた漢字かな交じり文
        returns:
            kanjikana_parse: MeCabによってパースした漢字かな交じり文
            kana_space: 空白で区切ったひらがな文
            phoneme: 空白で区切った音素列
        """
        kanjikana = self.normalize_japanese_text(kanjikana)
        kana = pyopenjtalk.g2p(kanjikana, kana=True)
        phoneme = pyopenjtalk.g2p(kanjikana)
        kanjikana_parse = self.mecab.parse(kanjikana).replace(" \n", "")
        kana_space = " ".join(kana)
        return kanjikana_parse, kana_space, phoneme

    def update_total_errors_dct(
        self,
        jiwer_result: jiwer.process.WordOutput,
        kind: str,
        error_type: str,
        utt_gt_len: int,
    ) -> None:
        delt = jiwer_result.deletions
        inst = jiwer_result.deletions
        subst = jiwer_result.substitutions
        self.total_errors_dct[kind][error_type].deletions += delt
        self.total_errors_dct[kind][error_type].insertions += inst
        self.total_errors_dct[kind][error_type].substitutions += subst
        self.total_errors_dct[kind][error_type].distance += delt + inst + subst
        self.total_errors_dct[kind][error_type].length += utt_gt_len

    def test_step_process(
        self,
        wav_eval: torch.Tensor,
        wav_gt: torch.Tensor,
        n_sample_min: int,
        utt_gt_kanjikana: str,
        speaker: str,
        filename: str,
        kind: str,
        wandb_table_data: list,
    ) -> None:
        """
        args:
            wav_eval: (B, T)
            wav_gt: (B, T)
            n_sample_min:
            n_sample_min:
            utt_gt_kanjikana:
            speaker:
            filename:
            kind:
            is_gt:
            wandb_table_data:
        """
        wav_eval = self.process_wav(wav_eval, n_sample_min)
        wav_gt = self.process_wav(wav_gt, n_sample_min)

        if kind == "gt":
            pesq, stoi, estoi = None, None, None
        else:
            pesq = self.wb_pesq_evaluator(wav_eval, wav_gt)
            stoi = self.stoi_evaluator(wav_eval, wav_gt)
            estoi = self.estoi_evaluator(wav_eval, wav_gt)

        utt_recog = self.speech_recognizer.transcribe(
            wav_eval.to(dtype=torch.float32),
            language="ja",
            temperature=(0.0),
        )
        utt_recog_kanjikana = utt_recog["text"]

        (
            utt_recog_kanjikana_processed,
            utt_recog_kana_processed,
            utt_recog_phoneme_processed,
        ) = self.preprocess_text_for_error_rate(utt_recog_kanjikana)
        (
            utt_gt_kanjikana_processed,
            utt_gt_kana_processed,
            utt_gt_phoneme_processed,
        ) = self.preprocess_text_for_error_rate(utt_gt_kanjikana)

        jiwer_result_kanjikana = jiwer.process_words(
            utt_gt_kanjikana_processed,
            utt_recog_kanjikana_processed,
        )
        jiwer_result_kana = jiwer.process_words(
            utt_gt_kana_processed,
            utt_recog_kana_processed,
        )
        jiwer_result_phoneme = jiwer.process_words(
            utt_gt_phoneme_processed,
            utt_recog_phoneme_processed,
        )

        self.update_total_errors_dct(
            jiwer_result_kanjikana,
            kind,
            "kanjikana",
            len(utt_gt_kanjikana_processed.split(" ")),
        )
        self.update_total_errors_dct(
            jiwer_result_kana,
            kind,
            "kana",
            len(utt_gt_kana_processed.split(" ")),
        )
        self.update_total_errors_dct(
            jiwer_result_phoneme,
            kind,
            "phoneme",
            len(utt_gt_phoneme_processed.split(" ")),
        )

        save_dir = (
            Path(
                str(Path(self.cfg.training.checkpoints_save_dir)).replace(
                    "checkpoints", "results"
                )
            )
            / speaker
            / filename
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        write(
            filename=str(save_dir / f"{kind}.wav"),
            rate=self.cfg.data.audio.sr,
            data=wav_eval.cpu().numpy(),
        )

        wandb_table_data.append(
            [
                speaker,
                filename,
                kind,
                wandb.Audio(wav_eval.cpu(), sample_rate=self.cfg.data.audio.sr),
                pesq,
                stoi,
                estoi,
                utt_gt_kanjikana_processed,
                utt_recog_kanjikana_processed,
                jiwer_result_kanjikana.deletions,
                jiwer_result_kanjikana.insertions,
                jiwer_result_kanjikana.substitutions,
                jiwer_result_kanjikana.wer,
                utt_gt_kana_processed,
                utt_recog_kana_processed,
                jiwer_result_kana.deletions,
                jiwer_result_kana.insertions,
                jiwer_result_kana.substitutions,
                jiwer_result_kana.wer,
                utt_gt_phoneme_processed,
                utt_recog_phoneme_processed,
                jiwer_result_phoneme.deletions,
                jiwer_result_phoneme.insertions,
                jiwer_result_phoneme.substitutions,
                jiwer_result_phoneme.wer,
            ]
        )

    def test_step(self, batch: list, batch_index: int) -> None:
        (
            wav_gt,
            lip,
            feature,
            hubert_conv_feature,
            hubert_layer_feature_cluster,
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
                pred_ssl_feature_cluster,
                pred_mel_speech_ssl,
                pred_ssl_feature_cluster_speech_ssl,
                pred_mel_ensemble,
                pred_ssl_feature_cluster_ensemble,
            ) = self.model(
                lip=lip,
                audio=None,
                spk_emb=spk_emb,
                padding_mask_lip=padding_mask_lip,
                padding_mask_speech_ssl=~padding_mask_feature_speech_ssl,
            )

        inputs_dict_gt = self.hifigan_mel_speech_ssl.prepare_inputs_dict(
            feature, hubert_layer_feature_cluster
        )

        if self.cfg.model.decoder.vocoder_input == "avhubert":
            inputs_dict_pred = self.hifigan_mel_speech_ssl.prepare_inputs_dict(
                pred_mel, pred_ssl_feature_cluster.argmax(dim=1)
            )
        elif self.cfg.model.decoder.vocoder_input == "speech_ssl":
            inputs_dict_pred = self.hifigan_mel_speech_ssl.prepare_inputs_dict(
                pred_mel_speech_ssl, pred_ssl_feature_cluster_speech_ssl.argmax(dim=1)
            )
        elif self.cfg.model.decoder.vocoder_input == "ensemble":
            inputs_dict_pred = self.hifigan_mel_speech_ssl.prepare_inputs_dict(
                pred_mel_ensemble, pred_ssl_feature_cluster_ensemble.argmax(dim=1)
            )

        with torch.no_grad():
            wav_pred_mel_speech_ssl = self.hifigan_mel_speech_ssl.gen(
                inputs_dict_pred, spk_emb
            )
            wav_abs_mel_speech_ssl = self.hifigan_mel_speech_ssl.gen(
                inputs_dict_gt, spk_emb
            )

        wav_eval_dct = {
            "gt": wav_gt,
            "pred_mel_speech_ssl": wav_pred_mel_speech_ssl,
            "abs_mel_speech_ssl": wav_abs_mel_speech_ssl,
        }

        n_sample_min = min(
            wav_gt.shape[-1],
            wav_abs_mel_speech_ssl.shape[-1],
            wav_pred_mel_speech_ssl.shape[-1],
        )

        for i, row in self.df_utt.iterrows():
            if str(row["utt_num"]) in filename[0]:
                utt_gt_kanjikana = row["text"]

        wandb_table_data = []
        for wav_eval_kind, wav_eval in wav_eval_dct.items():
            self.test_step_process(
                wav_eval=wav_eval,
                wav_gt=wav_gt,
                n_sample_min=n_sample_min,
                utt_gt_kanjikana=utt_gt_kanjikana,
                speaker=speaker[0],
                filename=filename[0],
                kind=wav_eval_kind,
                wandb_table_data=wandb_table_data,
            )

        self.test_data_list += wandb_table_data

    def on_test_end(self) -> None:
        save_dir = Path(
            str(Path(self.cfg.training.checkpoints_save_dir)).replace(
                "checkpoints", "results"
            )
        )

        subprocess.run(
            [
                "/home/minami/Resemblyzer/.venv/bin/python",
                "/home/minami/Resemblyzer/calc_cossim.py",
                "--inp_dir",
                str(save_dir),
                "--out_path",
                str(save_dir / "spk_sim.csv"),
            ],
        )
        df_spk_sim = pl.read_csv(str(save_dir / "spk_sim.csv"))
        for i in range(len(self.test_data_list)):
            speaker = self.test_data_list[i][0]
            filename = self.test_data_list[i][1]
            kind = self.test_data_list[i][2]
            spk_sim = df_spk_sim.filter(
                (pl.col("speaker") == speaker)
                & (pl.col("filename") == filename)
                & (pl.col("kind") == kind)
            )
            assert spk_sim.shape[0] == 1, "spk_sim should have only one value."
            self.test_data_list[i].append(spk_sim["score"][0])

        subprocess.run(
            [
                "/home/minami/UTMOS-demo/.venv/bin/python",
                "/home/minami/UTMOS-demo/predict.py",
                "--ckpt_path",
                "/home/minami/UTMOS-demo/epoch=3-step=7459.ckpt",
                "--mode",
                "predict_dir",
                "--inp_dir",
                str(save_dir),
                "--out_path",
                str(save_dir / "utmos.csv"),
            ],
        )
        df_utmos = pl.read_csv(str(save_dir / "utmos.csv"))
        for i in range(len(self.test_data_list)):
            speaker = self.test_data_list[i][0]
            filename = self.test_data_list[i][1]
            kind = self.test_data_list[i][2]
            utmos = df_utmos.filter(
                (pl.col("speaker") == speaker)
                & (pl.col("filename") == filename)
                & (pl.col("kind") == kind)
            )
            assert utmos.shape[0] == 1, "UTMOS should have only one value."
            self.test_data_list[i].append(utmos["score"][0])

        table = wandb.Table(columns=self.test_data_columns, data=self.test_data_list)
        wandb.log({"test_data": table})

        kind_index = self.test_data_columns.index("kind")
        pesq_index = self.test_data_columns.index("pesq")
        stoi_index = self.test_data_columns.index("stoi")
        estoi_index = self.test_data_columns.index("estoi")
        utmos_index = self.test_data_columns.index("utmos")
        spk_sim_index = self.test_data_columns.index("spk_sim")

        kinds = set()
        for i in range(len(self.test_data_list)):
            kinds.add(self.test_data_list[i][kind_index])

        result = {}
        for kind in kinds:
            result[kind] = {
                "pesq": [],
                "stoi": [],
                "estoi": [],
                "spk_sim": [],
                "utmos": [],
            }

        for test_data in self.test_data_list:
            kind = test_data[kind_index]
            pesq = test_data[pesq_index]
            stoi = test_data[stoi_index]
            estoi = test_data[estoi_index]
            utmos = test_data[utmos_index]
            spk_sim = test_data[spk_sim_index]
            result[kind]["pesq"].append(pesq)
            result[kind]["stoi"].append(stoi)
            result[kind]["estoi"].append(estoi)
            result[kind]["spk_sim"].append(spk_sim)
            result[kind]["utmos"].append(utmos)

        columns = [
            "kind",
            "pesq",
            "stoi",
            "estoi",
            "wer_kanjikana",
            "cer_kana",
            "per_phoneme",
            "spk_sim",
            "utmos",
        ]
        data_list = []
        for kind, value_dict in result.items():
            if kind == "gt":
                data_list.append(
                    [
                        kind,
                        None,
                        None,
                        None,
                        self.total_errors_dct[kind]["kanjikana"].error_rate,
                        self.total_errors_dct[kind]["kana"].error_rate,
                        self.total_errors_dct[kind]["phoneme"].error_rate,
                        np.mean(value_dict["spk_sim"]),
                        np.mean(value_dict["utmos"]),
                    ]
                )
            else:
                data_list.append(
                    [
                        kind,
                        np.mean(value_dict["pesq"]),
                        np.mean(value_dict["stoi"]),
                        np.mean(value_dict["estoi"]),
                        self.total_errors_dct[kind]["kanjikana"].error_rate,
                        self.total_errors_dct[kind]["kana"].error_rate,
                        self.total_errors_dct[kind]["phoneme"].error_rate,
                        np.mean(value_dict["spk_sim"]),
                        np.mean(value_dict["utmos"]),
                    ]
                )
        table = wandb.Table(columns=columns, data=data_list)
        wandb.log({"metrics_mean": table})

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
