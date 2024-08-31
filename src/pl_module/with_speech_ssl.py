import logging
from pathlib import Path

import lightning as L
import MeCab
import numpy as np
import omegaconf
import pandas as pd
import torch
import whisper
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from jiwer import wer
from scipy.io.wavfile import write
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

import wandb
from src.data_process.utils import get_upsample, get_upsample_speech_ssl
from src.log_fn.save_loss import save_epoch_loss_plot
from src.log_fn.save_sample import save_mel
from src.loss_fn.base import LossFunctions
from src.model.with_speech_ssl import WithSpeechSSLModel
from src.pl_module.hifigan_multiple_ssl import LitHiFiGANMultipltSSLModel

logger = logging.getLogger(__name__)


class LitWithSpeechSSLModule(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.automatic_optimization = True

        self.model = WithSpeechSSLModel(cfg)

        if cfg.model.avhubert.freeze:
            logger.info("freeze avhubert parameters.")
            for param in self.model.avhubert.parameters():
                param.requires_grad = False

        if cfg.model.spk_emb_layer.freeze:
            logger.info("freeze spk_emb_layer parameters.")
            for param in self.model.spk_emb_layer.parameters():
                param.requires_grad = False

        if cfg.model.decoder.conv.freeze:
            logger.info("freeze mel_decoder parameters.")
            for param in self.model.mel_decoder.parameters():
                param.requires_grad = False

        if cfg.model.decoder.linear.freeze:
            logger.info("freeze ssl_feature_cluster_decoder_linear parameters.")
            for param in self.model.ssl_feature_cluster_decoder_linear.parameters():
                param.requires_grad = False

        if cfg.model.decoder.speech_ssl.freeze:
            logger.info("freeze ssl_feature_cluster_decoder_ssl parameters.")
            for param in self.model.ssl_feature_cluster_decoder_ssl.parameters():
                param.requires_grad = False

        if (
            not cfg.model.decoder.speech_ssl.freeze
        ) and cfg.model.decoder.speech_ssl.partial_update.use:
            logger.info("Freeze Some Speech SSL Layers for Partial Update")
            for layer_index, layer in enumerate(
                self.model.ssl_feature_cluster_decoder_ssl.ssl_model_encoder.layers
            ):
                if (
                    layer_index
                    < self.cfg.model.decoder.speech_ssl.partial_update.update_layer_index_lower
                ):
                    logger.info(f"freeze: {layer_index}")
                    for param in layer.parameters():
                        param.requires_grad = False

        self.loss_fn = LossFunctions()

        self.train_step_mel_loss_list = []
        self.train_step_ssl_conv_feature_loss_list = []
        self.train_step_ssl_feature_cluster_linear_loss_list = []
        self.train_step_ssl_feature_cluster_ssl_loss_list = []
        self.train_step_total_loss_list = []
        self.train_epoch_mel_loss_list = []
        self.train_epoch_ssl_conv_feature_loss_list = []
        self.train_epoch_ssl_feature_cluster_linear_loss_list = []
        self.train_epoch_ssl_feature_cluster_ssl_loss_list = []
        self.train_epoch_total_loss_list = []
        self.val_step_mel_loss_list = []
        self.val_step_ssl_conv_feature_loss_list = []
        self.val_step_ssl_feature_cluster_linear_loss_list = []
        self.val_step_ssl_feature_cluster_ssl_loss_list = []
        self.val_step_total_loss_list = []
        self.val_epoch_mel_loss_list = []
        self.val_epoch_ssl_conv_feature_loss_list = []
        self.val_epoch_ssl_feature_cluster_linear_loss_list = []
        self.val_epoch_ssl_feature_cluster_ssl_loss_list = []
        self.val_epoch_total_loss_list = []
        self.train_mel_example = {"gt": None, "pred": None}
        self.val_mel_example = {"gt": None, "pred": None}

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

        (
            pred_mel,
            pred_ssl_conv_feature,
            pred_ssl_feature_cluster_linear,
            pred_ssl_feature_cluster_ssl,
        ) = self.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            padding_mask_lip=padding_mask_lip,
            padding_mask_feature=padding_mask_lip_inverse,
        )

        mel_loss = self.loss_fn.l1_loss(
            pred=pred_mel, target=feature, mask=padding_mask_feature
        )

        if self.cfg.model.decoder.speech_ssl.model_name == "rinna/japanese-hubert-base":
            ssl_conv_feature_loss = self.loss_fn.l1_loss(
                pred=pred_ssl_conv_feature,
                target=hubert_conv_feature,
                mask=padding_mask_feature_speech_ssl,
            )
            ssl_feature_cluster_linear_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_linear,
                target=hubert_final_feature_cluster,
                ignore_index=0,
            )
            ssl_feature_cluster_ssl_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_ssl,
                target=hubert_final_feature_cluster,
                ignore_index=0,
            )
        elif (
            self.cfg.model.decoder.speech_ssl.model_name
            == "rinna/japanese-wav2vec2-base"
        ):
            ssl_conv_feature_loss = self.loss_fn.l1_loss(
                pred=pred_ssl_conv_feature,
                target=wav2vec2_conv_feature,
                mask=padding_mask_feature_speech_ssl,
            )
            ssl_feature_cluster_linear_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_linear,
                target=wav2vec2_final_feature_cluster,
                ignore_index=0,
            )
            ssl_feature_cluster_ssl_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_ssl,
                target=wav2vec2_final_feature_cluster,
                ignore_index=0,
            )
        elif (
            self.cfg.model.decoder.speech_ssl.model_name
            == "rinna/japanese-data2vec-audio-base"
        ):
            ssl_conv_feature_loss = self.loss_fn.l1_loss(
                pred=pred_ssl_conv_feature,
                target=data2vec_conv_feature,
                mask=padding_mask_feature_speech_ssl,
            )
            ssl_feature_cluster_linear_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_linear,
                target=data2vec_final_feature_cluster,
                ignore_index=0,
            )
            ssl_feature_cluster_ssl_loss = torch.nn.functional.cross_entropy(
                input=pred_ssl_feature_cluster_ssl,
                target=data2vec_final_feature_cluster,
                ignore_index=0,
            )

        total_loss = (
            (mel_loss * self.cfg.training.loss_weights.mel_loss)
            + (
                ssl_conv_feature_loss
                * self.cfg.training.loss_weights.ssl_conv_feature_loss
            )
            + (
                ssl_feature_cluster_linear_loss
                * self.cfg.training.loss_weights.ssl_feature_cluster_linear_loss
            )
            + (
                ssl_feature_cluster_ssl_loss
                * self.cfg.training.loss_weights.ssl_feature_cluster_ssl_loss
            )
        )

        return (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_linear_loss,
            ssl_feature_cluster_ssl_loss,
            total_loss,
            feature,
            pred_mel,
        )

    def training_step(self, batch: list, batch_index: int) -> torch.Tensor:
        (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_linear_loss,
            ssl_feature_cluster_ssl_loss,
            total_loss,
            feature,
            pred_mel,
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
            "train_ssl_feature_cluster_linear_loss",
            ssl_feature_cluster_linear_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_ssl_feature_cluster_ssl_loss",
            ssl_feature_cluster_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "train_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.train_step_mel_loss_list.append(mel_loss.item())
        self.train_step_ssl_conv_feature_loss_list.append(ssl_conv_feature_loss.item())
        self.train_step_ssl_feature_cluster_linear_loss_list.append(
            ssl_feature_cluster_linear_loss.item()
        )
        self.train_step_ssl_feature_cluster_ssl_loss_list.append(
            ssl_feature_cluster_ssl_loss.item()
        )
        self.train_step_total_loss_list.append(total_loss.item())

        self.train_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.train_mel_example["pred"] = (
            pred_mel[0].cpu().detach().numpy().astype(np.float32)
        )

        return total_loss

    def validation_step(self, batch: list, batch_index: int) -> None:
        (
            mel_loss,
            ssl_conv_feature_loss,
            ssl_feature_cluster_linear_loss,
            ssl_feature_cluster_ssl_loss,
            total_loss,
            feature,
            pred_mel,
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
            "val_ssl_feature_cluster_linear_loss",
            ssl_feature_cluster_linear_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_ssl_feature_cluster_ssl_loss",
            ssl_feature_cluster_ssl_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )
        self.log(
            "val_total_loss",
            total_loss,
            logger=True,
            batch_size=self.cfg.training.batch_size,
        )

        self.val_step_mel_loss_list.append(mel_loss.item())
        self.val_step_ssl_conv_feature_loss_list.append(ssl_conv_feature_loss.item())
        self.val_step_ssl_feature_cluster_linear_loss_list.append(
            ssl_feature_cluster_linear_loss.item()
        )
        self.val_step_ssl_feature_cluster_ssl_loss_list.append(
            ssl_feature_cluster_ssl_loss.item()
        )
        self.val_step_total_loss_list.append(total_loss.item())

        self.val_mel_example["gt"] = (
            feature[0].cpu().detach().numpy().astype(np.float32)
        )
        self.val_mel_example["pred"] = (
            pred_mel[0].cpu().detach().numpy().astype(np.float32)
        )

    def on_validation_epoch_end(self) -> None:
        self.train_epoch_mel_loss_list.append(np.mean(self.train_step_mel_loss_list))
        self.train_epoch_ssl_conv_feature_loss_list.append(
            np.mean(self.train_step_ssl_conv_feature_loss_list)
        )
        self.train_epoch_ssl_feature_cluster_linear_loss_list.append(
            np.mean(self.train_step_ssl_feature_cluster_linear_loss_list)
        )
        self.train_epoch_ssl_feature_cluster_ssl_loss_list.append(
            np.mean(self.train_step_ssl_feature_cluster_ssl_loss_list)
        )
        self.train_epoch_total_loss_list.append(
            np.mean(self.train_step_total_loss_list)
        )
        self.train_step_mel_loss_list.clear()
        self.train_step_ssl_conv_feature_loss_list.clear()
        self.train_step_ssl_feature_cluster_linear_loss_list.clear()
        self.train_step_ssl_feature_cluster_ssl_loss_list.clear()
        self.train_step_total_loss_list.clear()

        self.val_epoch_mel_loss_list.append(np.mean(self.val_step_mel_loss_list))
        self.val_epoch_ssl_conv_feature_loss_list.append(
            np.mean(self.val_step_ssl_conv_feature_loss_list)
        )
        self.val_epoch_ssl_feature_cluster_linear_loss_list.append(
            np.mean(self.val_step_ssl_feature_cluster_linear_loss_list)
        )
        self.val_epoch_ssl_feature_cluster_ssl_loss_list.append(
            np.mean(self.val_step_ssl_feature_cluster_ssl_loss_list)
        )
        self.val_epoch_total_loss_list.append(np.mean(self.val_step_total_loss_list))
        self.val_step_mel_loss_list.clear()
        self.val_step_ssl_conv_feature_loss_list.clear()
        self.val_step_ssl_feature_cluster_linear_loss_list.clear()
        self.val_step_ssl_feature_cluster_ssl_loss_list.clear()
        self.val_step_total_loss_list.clear()

        save_epoch_loss_plot(
            title="mel_loss",
            train_loss_list=self.train_epoch_mel_loss_list,
            val_loss_list=self.val_epoch_mel_loss_list,
        )
        save_epoch_loss_plot(
            title="ssl_conv_feature_loss",
            train_loss_list=self.train_epoch_ssl_conv_feature_loss_list,
            val_loss_list=self.val_epoch_ssl_conv_feature_loss_list,
        )
        save_epoch_loss_plot(
            title="ssl_feature_cluster_linear_loss",
            train_loss_list=self.train_epoch_ssl_feature_cluster_linear_loss_list,
            val_loss_list=self.val_epoch_ssl_feature_cluster_linear_loss_list,
        )
        save_epoch_loss_plot(
            title="ssl_feature_cluster_ssl_loss",
            train_loss_list=self.train_epoch_ssl_feature_cluster_ssl_loss_list,
            val_loss_list=self.val_epoch_ssl_feature_cluster_ssl_loss_list,
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
        ) = self.model(
            lip=lip,
            audio=None,
            spk_emb=spk_emb,
            padding_mask_lip=padding_mask_lip,
            padding_mask_feature=padding_mask_lip,
        )

        if self.cfg.model.decoder.vocoder_input_cluster == "conv":
            inputs_dict = self.hifigan.prepare_inputs_dict(
                feature=pred_mel,
                feature_hubert_encoder=None,
                feature_hubert_cluster=pred_ssl_feature_cluster_ssl.argmax(dim=1),
            )
        elif self.cfg.model.decoder.vocoder_input_cluster == "hubert":
            inputs_dict = self.hifigan.prepare_inputs_dict(
                feature=pred_mel,
                feature_hubert_encoder=None,
                feature_hubert_cluster=pred_ssl_feature_cluster_ssl.argmax(dim=1),
            )
        wav_pred = self.hifigan.gen(inputs_dict, spk_emb)

        inputs_dict = self.hifigan.prepare_inputs_dict(
            feature=feature,
            feature_hubert_encoder=None,
            feature_hubert_cluster=hubert_final_feature_cluster,
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
        self.model.eval()

        self.hifigan = LitHiFiGANMultipltSSLModel.load_from_checkpoint(
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
