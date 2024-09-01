from pathlib import Path

import librosa
import numpy as np
import omegaconf
import torch
import torchvision
from torch.utils.data import Dataset

from src.data_process.utils import (
    get_upsample,
    get_upsample_speech_ssl,
    wav2mel,
)
from src.dataset.utils import (
    get_spk_emb,
    get_spk_emb_hifi_captain,
    get_spk_emb_jsut,
    get_spk_emb_jvs,
)
from src.transform.with_speech_ssl import WithSpeechSSLTransform


class WithSpeechSSLDataset(Dataset):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        data_path_list: list[dict[str, Path]],
        transform: WithSpeechSSLTransform,
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
        hubert_final_feature_path = self.data_path_list[index][
            "hubert_final_feature_path"
        ]
        hubert_final_feature_cluster_path = self.data_path_list[index][
            "hubert_final_feature_cluster_path"
        ]
        wav2vec2_conv_feature_path = self.data_path_list[index][
            "wav2vec2_conv_feature_path"
        ]
        wav2vec2_final_feature_path = self.data_path_list[index][
            "wav2vec2_final_feature_path"
        ]
        wav2vec2_final_feature_cluster_path = self.data_path_list[index][
            "wav2vec2_final_feature_cluster_path"
        ]
        data2vec_conv_feature_path = self.data_path_list[index][
            "data2vec_conv_feature_path"
        ]
        data2vec_final_feature_path = self.data_path_list[index][
            "data2vec_final_feature_path"
        ]
        data2vec_final_feature_cluster_path = self.data_path_list[index][
            "data2vec_final_feature_cluster_path"
        ]
        speaker = self.data_path_list[index]["speaker"]
        filename = self.data_path_list[index]["filename"]
        spk_emb = torch.from_numpy(self.embs[speaker]).to(torch.float32)

        wav, _ = librosa.load(str(audio_path), sr=self.cfg.data.audio.sr)
        wav = wav / np.max(np.abs(wav))  # (T,)
        feature = wav2mel(wav, self.cfg, ref_max=False)  # (C, T)
        hubert_conv_feature = np.load(str(hubert_conv_feature_path)).T
        hubert_final_feature = np.load(str(hubert_final_feature_path)).T
        hubert_final_feature_cluster = np.load(str(hubert_final_feature_cluster_path))
        wav2vec2_conv_feature = np.load(str(wav2vec2_conv_feature_path)).T
        wav2vec2_final_feature = np.load(str(wav2vec2_final_feature_path)).T
        wav2vec2_final_feature_cluster = np.load(
            str(wav2vec2_final_feature_cluster_path)
        )
        data2vec_conv_feature = np.load(str(data2vec_conv_feature_path)).T
        data2vec_final_feature = np.load(str(data2vec_final_feature_path)).T
        data2vec_final_feature_cluster = np.load(
            str(data2vec_final_feature_cluster_path)
        )
        upsample = get_upsample(self.cfg)
        upsample_speech_ssl = get_upsample_speech_ssl(self.cfg)

        if video_path is not None:
            lip, _, _ = torchvision.io.read_video(
                str(video_path), pts_unit="sec", output_format="TCHW"
            )  # (T, C, H, W)
            lip = torchvision.transforms.functional.rgb_to_grayscale(lip)
        else:
            lip = torch.rand(int(feature.shape[1] * upsample), 1, 96, 96)
        lip = lip.numpy()

        lip_len = min(
            int(lip.shape[0]),
            int(feature.shape[1] // upsample),
            int(hubert_conv_feature.shape[1] // upsample_speech_ssl),
            int(hubert_final_feature.shape[1] // upsample_speech_ssl),
            int(hubert_final_feature_cluster.shape[0] // upsample_speech_ssl),
            int(wav2vec2_conv_feature.shape[1] // upsample_speech_ssl),
            int(wav2vec2_final_feature.shape[1] // upsample_speech_ssl),
            int(wav2vec2_final_feature_cluster.shape[0] // upsample_speech_ssl),
            int(data2vec_conv_feature.shape[1] // upsample_speech_ssl),
            int(data2vec_final_feature.shape[1] // upsample_speech_ssl),
            int(data2vec_final_feature_cluster.shape[0] // upsample_speech_ssl),
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
        hubert_final_feature = self.adjust_sequence_length(
            hubert_final_feature,
            int(lip_len * upsample_speech_ssl),
        )
        hubert_final_feature_cluster = self.adjust_sequence_length(
            hubert_final_feature_cluster,
            int(lip_len * upsample_speech_ssl),
        )
        wav2vec2_conv_feature = self.adjust_sequence_length(
            wav2vec2_conv_feature,
            int(lip_len * upsample_speech_ssl),
        )
        wav2vec2_final_feature = self.adjust_sequence_length(
            wav2vec2_final_feature,
            int(lip_len * upsample_speech_ssl),
        )
        wav2vec2_final_feature_cluster = self.adjust_sequence_length(
            wav2vec2_final_feature_cluster,
            int(lip_len * upsample_speech_ssl),
        )
        data2vec_conv_feature = self.adjust_sequence_length(
            data2vec_conv_feature,
            int(lip_len * upsample_speech_ssl),
        )
        data2vec_final_feature = self.adjust_sequence_length(
            data2vec_final_feature,
            int(lip_len * upsample_speech_ssl),
        )
        data2vec_final_feature_cluster = self.adjust_sequence_length(
            data2vec_final_feature_cluster,
            int(lip_len * upsample_speech_ssl),
        )
        lip = self.adjust_sequence_length(lip, lip_len)

        wav = torch.from_numpy(wav).to(torch.float32)
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0).to(torch.float32)
        feature = torch.from_numpy(feature).to(torch.float32).permute(1, 0)
        hubert_conv_feature = torch.from_numpy(hubert_conv_feature).to(torch.float32)
        hubert_final_feature = torch.from_numpy(hubert_final_feature).to(torch.float32)
        hubert_final_feature_cluster = torch.from_numpy(
            hubert_final_feature_cluster
        ).to(torch.long)
        wav2vec2_conv_feature = torch.from_numpy(wav2vec2_conv_feature).to(
            torch.float32
        )
        wav2vec2_final_feature = torch.from_numpy(wav2vec2_final_feature).to(
            torch.float32
        )
        wav2vec2_final_feature_cluster = torch.from_numpy(
            wav2vec2_final_feature_cluster
        ).to(torch.long)
        data2vec_conv_feature = torch.from_numpy(data2vec_conv_feature).to(
            torch.float32
        )
        data2vec_final_feature = torch.from_numpy(data2vec_final_feature).to(
            torch.float32
        )
        data2vec_final_feature_cluster = torch.from_numpy(
            data2vec_final_feature_cluster
        ).to(torch.long)

        lip, feature = self.transform(
            lip=lip,
            feature=feature,
            lip_mean=self.lip_mean,
            lip_std=self.lip_std,
        )

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)

        feature_len = torch.tensor(feature.shape[-1]).to(torch.int)
        feature_ssl_len = torch.tensor(hubert_conv_feature.shape[-1]).to(torch.int)
        lip_len = torch.tensor(lip.shape[-1]).to(torch.int)

        return (
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
        )
