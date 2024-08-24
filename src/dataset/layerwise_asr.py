import pathlib

import librosa
import numpy as np
import omegaconf
import pyopenjtalk
import torch
from torch.utils.data import Dataset

from src.dataset.utils import get_token_index_mapping


class LayerwiseASRDataset(Dataset):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        data_path_list: list[dict[str, pathlib.Path]],
        transform=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_path_list = data_path_list
        self.transform = transform
        self.token_to_id, self.id_to_token = get_token_index_mapping(cfg)

    def __len__(self) -> int:
        return len(self.data_path_list)

    def __getitem__(self, index: int) -> tuple:
        audio_path = self.data_path_list[index]["audio_path"]
        text_path = self.data_path_list[index]["text_path"]
        speaker = self.data_path_list[index]["speaker"]
        filename = self.data_path_list[index]["filename"]

        wav, _ = librosa.load(str(audio_path), sr=self.cfg.data.audio.sr)
        wav = wav / np.max(np.abs(wav))  # (T,)
        wav = wav[: wav.shape[0] - (wav.shape[0] % (self.cfg.data.audio.sr // 100))]
        wav = np.pad(
            wav,
            (0, int(self.cfg.data.audio.sr // 100 * 2)),
            mode="constant",
            constant_values=0,
        )
        wav = torch.from_numpy(wav).to(torch.float32)  # (T,)
        wav_len = wav.shape[0]

        feature_len = int(wav_len / self.cfg.data.audio.sr * 50)

        with open(str(text_path), "r", encoding="utf-8") as file:
            text = file.read()
        token = pyopenjtalk.g2p(text)
        token = token.split(" ")
        # token.insert(0, "^")
        # token.append("$")
        token = torch.tensor([self.token_to_id[t] for t in token])
        token_len = token.shape[0]

        return (
            wav,
            wav_len,
            feature_len,
            token,
            token_len,
            speaker,
            filename,
        )
