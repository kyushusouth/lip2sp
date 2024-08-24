import functools
from pathlib import Path

import lightning as L
import omegaconf
import pandas as pd
from torch.utils.data import DataLoader

from src.collate_fn.layerwise_asr import adjust_seq_lengths
from src.dataset.layerwise_asr import LayerwiseASRDataset


class LayerwiseASRDataModule(L.LightningDataModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size

    def get_kablab_path_list(self, df: pd.DataFrame, data_split: str) -> list:
        df = df.loc[df["data_split"] == data_split]
        audio_dir = Path(self.cfg.path.kablab.audio_dir).expanduser()
        text_dir = Path(self.cfg.path.kablab.text_dir).expanduser()
        data_path_list = []
        for i, row in df.iterrows():
            audio_path = audio_dir / row["speaker"] / f'{row["filename"]}.wav'
            text_path = text_dir / f'{row["filename"]}.txt'
            if not audio_path.exists() or not text_path.exists():
                continue
            data_path_list.append(
                {
                    "audio_path": audio_path,
                    "text_path": text_path,
                    "speaker": row["speaker"],
                    "filename": row["filename"],
                }
            )
        return data_path_list

    def setup(self, stage: str) -> None:
        train_data_path_list = []
        val_data_path_list = []
        test_data_path_list = []

        df = pd.read_csv(str(Path(self.cfg.path.kablab.df_path).expanduser()))
        df = df.loc[df["speaker"].isin(self.cfg.data_choice.kablab.speaker)]
        df = df.loc[df["corpus"].isin(self.cfg.data_choice.kablab.corpus)]
        train_data_path_list += self.get_kablab_path_list(df, "train")
        val_data_path_list += self.get_kablab_path_list(df, "val")
        test_data_path_list += self.get_kablab_path_list(df, "test")

        if stage == "fit":
            self.train_dataset = LayerwiseASRDataset(
                cfg=self.cfg,
                data_path_list=train_data_path_list,
                transform=None,
            )
            self.val_dataset = LayerwiseASRDataset(
                cfg=self.cfg,
                data_path_list=val_data_path_list,
                transform=None,
            )
        if stage == "test":
            self.test_dataset = LayerwiseASRDataset(
                cfg=self.cfg,
                data_path_list=test_data_path_list,
                transform=None,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=functools.partial(adjust_seq_lengths, cfg=self.cfg),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=functools.partial(adjust_seq_lengths, cfg=self.cfg),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=None,
        )
