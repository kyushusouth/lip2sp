import logging
from pathlib import Path

import librosa
import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from src.data_process.utils import (
    get_upsample,
    get_upsample_speech_ssl,
    load_video,
    wav2mel,
)
from src.dataset.utils import (
    get_spk_emb,
    get_spk_emb_hifi_captain,
    get_spk_emb_jsut,
    get_spk_emb_jvs,
)
from src.transform.base_hubert_2 import BaseHuBERT2Transform

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseHuBERT2Dataset(Dataset):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        data_path_list: list[dict[str, Path]],
        transform: BaseHuBERT2Transform,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_path_list = data_path_list
        self.transform = transform
        self.embs = get_spk_emb(cfg)
        self.embs.update(get_spk_emb_jvs(cfg))
        self.embs.update(get_spk_emb_hifi_captain(cfg))
        self.embs.update(get_spk_emb_jsut(cfg))
        self.lip_mean = torch.from_numpy(np.array([cfg.data.video.lip_mean]))
        self.lip_std = torch.from_numpy(np.array([cfg.data.video.lip_std]))

    def __len__(self) -> int:
        return len(self.data_path_list)

    def adjust_sequence_length(self, x: np.ndarray, max_len: int) -> np.ndarray:
        if len(x.shape) == 1:
            x = x[:max_len]
            x_padded = np.zeros((max_len,))
            x_padded[: x.shape[0]] = x
        elif len(x.shape) == 2:
            x = x[:, :max_len]
            x_padded = np.zeros((x.shape[0], max_len))
            x_padded[:, : x.shape[1]] = x
        elif len(x.shape) == 4:
            x = x[:max_len]
            x_padded = np.zeros((max_len, x.shape[1], x.shape[2], x.shape[3]))
            x_padded[: x.shape[0]] = x
        return x_padded

    def __getitem__(self, index: int) -> tuple:
        audio_path = self.data_path_list[index]["audio_path"]
        video_path = self.data_path_list[index]["video_path"]
        hubert_conv_feature_path = self.data_path_list[index][
            "hubert_conv_feature_path"
        ]
        hubert_layer_feature_cluster_path = self.data_path_list[index][
            "hubert_layer_feature_cluster_path"
        ]
        speaker = self.data_path_list[index]["speaker"]
        filename = self.data_path_list[index]["filename"]
        spk_emb = torch.from_numpy(self.embs[speaker]).to(torch.float32)

        wav, _ = librosa.load(str(audio_path), sr=self.cfg.data.audio.sr)
        wav = wav / np.max(np.abs(wav))  # (T,)
        feature = wav2mel(wav, self.cfg, ref_max=False)  # (C, T)
        hubert_conv_feature = np.load(str(hubert_conv_feature_path)).T  # (C, T)
        hubert_layer_feature_cluster = np.load(str(hubert_layer_feature_cluster_path))
        upsample = get_upsample(self.cfg)
        upsample_speech_ssl = get_upsample_speech_ssl(self.cfg)

        if video_path is not None:
            lip = load_video(str(video_path))
            if lip is None:
                raise ValueError("Lip was None.")
            lip = np.expand_dims(lip, 1)  # (T, 1, H, W)
            # lip, _, _ = torchvision.io.read_video(
            #     str(video_path), pts_unit="sec", output_format="TCHW"
            # )  # (T, C, H, W)
            # lip = torchvision.transforms.functional.rgb_to_grayscale(lip)
        else:
            lip = np.random.rand(
                feature.shape[1] // upsample,
                1,
                self.cfg.data.video.imsize,
                self.cfg.data.video.imsize,
            )
            # lip = torch.rand(feature.shape[1] // upsample, 1, 96, 96)
        # lip = lip.numpy()

        logger.debug(
            f"Before adjusting sequence length: {lip.shape=}, {feature.shape=}, {hubert_conv_feature.shape=}, {hubert_layer_feature_cluster.shape=}"
        )

        lip_len = min(
            int(lip.shape[0]),
            int(feature.shape[1] // upsample),
            int(hubert_conv_feature.shape[1] // upsample_speech_ssl),
            int(hubert_layer_feature_cluster.shape[0] // upsample_speech_ssl),
        )

        wav = self.adjust_sequence_length(
            wav,
            int(self.cfg.data.audio.hop_length * upsample * lip_len),
        )
        feature = self.adjust_sequence_length(
            feature,
            int(lip_len * upsample),
        )
        hubert_conv_feature = self.adjust_sequence_length(
            hubert_conv_feature,
            int(lip_len * upsample_speech_ssl),
        )
        hubert_layer_feature_cluster = self.adjust_sequence_length(
            hubert_layer_feature_cluster,
            int(lip_len * upsample_speech_ssl),
        )
        lip = self.adjust_sequence_length(lip, lip_len)

        wav = torch.from_numpy(wav).to(torch.float32)
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0).to(torch.float32)
        feature = torch.from_numpy(feature).to(torch.float32).permute(1, 0)
        hubert_conv_feature = torch.from_numpy(hubert_conv_feature).to(torch.float32)
        hubert_layer_feature_cluster = torch.from_numpy(
            hubert_layer_feature_cluster
        ).to(torch.long)

        lip, feature = self.transform(
            lip=lip,
            feature=feature,
            lip_mean=self.lip_mean,
            lip_std=self.lip_std,
        )

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)

        logger.debug(
            f"After adjusting sequence length: {lip.shape=}, {feature.shape=}, {hubert_conv_feature.shape=}, {hubert_layer_feature_cluster.shape=}"
        )

        feature_len = torch.tensor(feature.shape[-1]).to(torch.int)
        feature_ssl_len = torch.tensor(hubert_layer_feature_cluster.shape[-1]).to(
            torch.int
        )
        lip_len = torch.tensor(lip.shape[-1]).to(torch.int)

        return (
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
        )
