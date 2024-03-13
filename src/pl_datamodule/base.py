from pathlib import Path

import lightning as L
import omegaconf
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset.base import BaseDataset


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform = None
        self.batch_size = None

    def get_path_list(self, df: pd.DataFrame, data_split: str) -> list:
        df = df.loc[df["data_split"] == data_split]
        audio_dir = Path(self.cfg["path"]["kablab"]["audio_dir"]).expanduser()
        video_dir = Path(self.cfg["path"]["kablab"]["video_dir"]).expanduser()
        data_path_list = []
        for i in range(df.shape[0]):
            row = df.iloc[i]
            audio_path = audio_dir / row["speaker"] / f'{row["filename"]}.wav'
            video_path = video_dir / row["speaker"] / f'{row["filename"]}.mp4'
            if (not audio_path.exists()) or (not video_path.exists()):
                continue
            data_path_list.append(
                {
                    "audio_path": audio_path,
                    "video_path": video_path,
                    "speaker": row["speaker"],
                    "filename": row["filename"],
                }
            )
        return data_path_list

    def setup(self, stage: str) -> None:
        """
        データのパス取得
        データセット定義
        """
        df = pd.read_csv(str(Path(self.cfg["path"]["kablab"]["df_path"]).expanduser()))
        df = df.loc[
            df["speaker"].isin(self.cfg["data_choice"]["kablab"]["speaker"])
        ]
        df = df.loc[df["corpus"].isin(self.cfg["data_choice"]["kablab"]["corpus"])]

        if stage == "fit":
            train_data_path_list = self.get_path_list(df, "train")
            val_data_path_list = self.get_path_list(df, "val")
            self.train_dataset = BaseDataset(
                cfg=self.cfg,
                data_path_list=train_data_path_list,
                transform=self.transform,
            )
            self.val_dataset = BaseDataset(
                cfg=self.cfg,
                data_path_list=val_data_path_list,
                transform=self.transform,
            )
        if stage == "test":
            test_data_path_list = self.get_path_list(df, "test")
            self.test_dataset = BaseDataset(
                cfg=self.cfg,
                data_path_list=test_data_path_list,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg['training']['num_workers'],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg['training']['num_workers'],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg['training']['num_workers'],
        )
