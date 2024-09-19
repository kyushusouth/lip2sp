import functools
import logging
import logging.config
from pathlib import Path

import lightning as L
import omegaconf
import pandas as pd
from torch.utils.data import DataLoader

from src.collate_fn.speech_memory import adjust_seq_lengths
from src.dataset.speech_memory import SpeechMemoryDataset
from src.transform.speech_memory import SpeechMemoryTransform

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpeechMemoryDataModule(L.LightningDataModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size

    def extend_path(
        self, process_data: str, dirpath: Path, row: pd.Series, suffix: str
    ) -> Path:
        if process_data == "kablab":
            path = dirpath / row["speaker"] / f'{row["filename"]}{suffix}'
        elif process_data == "hifi_captain":
            path = (
                dirpath
                / row["speaker"]
                / "wav"
                / row["parent_dir"]
                / f'{row["filename"]}{suffix}'
            )
        elif process_data == "jvs":
            path = (
                dirpath
                / row["speaker"]
                / row["data"]
                / "wav24kHz16bit"
                / f'{row["filename"]}{suffix}'
            )
        return path

    def get_path_list(
        self, df: pd.DataFrame, process_data: str, data_split: str
    ) -> list[dict[str, Path]]:
        df = df.loc[df["data_split"] == data_split]
        audio_dir = Path(self.cfg.path[process_data].audio_dir).expanduser()
        if process_data == "kablab":
            video_dir = Path(self.cfg.path[process_data].video_dir).expanduser()
        else:
            video_dir = None
        hubert_layer_feature_cluster_dir = (
            Path(
                self.cfg.path[process_data].hubert.layer_feature_cluster_dir
            ).expanduser()
            / self.cfg.model.decoder.speech_ssl.kmeans
            / str(self.cfg.model.decoder.speech_ssl.layer_index_cluster)
            / str(self.cfg.model.decoder.speech_ssl.n_clusters)
        )

        data_path_list = []
        for i, row in df.iterrows():
            audio_path = self.extend_path(process_data, audio_dir, row, ".wav")
            if process_data == "kablab":
                video_path = self.extend_path(process_data, video_dir, row, ".mp4")
            else:
                video_path = None
            hubert_layer_feature_cluster_path = self.extend_path(
                process_data, hubert_layer_feature_cluster_dir, row, ".npy"
            )

            if (
                (not audio_path.exists())
                or (not hubert_layer_feature_cluster_path.exists())
                or (process_data == "kablab" and (not video_path.exists()))
            ):
                continue

            data_path_list.append(
                {
                    "audio_path": audio_path,
                    "video_path": video_path,
                    "hubert_layer_feature_cluster_path": hubert_layer_feature_cluster_path,
                    "speaker": row["speaker"],
                    "filename": row["filename"],
                }
            )
        return data_path_list

    def setup(self, stage: str) -> None:
        logger.info("setup")
        train_data_path_list = []
        val_data_path_list = []
        test_data_path_list = []
        if self.cfg.data_choice.kablab.use:
            df = pd.read_csv(str(Path(self.cfg.path.kablab.df_path).expanduser()))
            df = df.loc[df["speaker"].isin(self.cfg.data_choice.kablab.speaker)]
            df = df.loc[df["corpus"].isin(self.cfg.data_choice.kablab.corpus)]
            train_data_path_list += self.get_path_list(df, "kablab", "train")
            val_data_path_list += self.get_path_list(df, "kablab", "val")
            test_data_path_list += self.get_path_list(df, "kablab", "test")
            logger.info(
                f"""
                use kablab
                {len(train_data_path_list)=}, {len(val_data_path_list)=}, {len(test_data_path_list)=}
                """
            )
        if self.cfg.data_choice.hifi_captain.use:
            df = pd.read_csv(str(Path(self.cfg.path.hifi_captain.df_path).expanduser()))
            train_data_path_list += self.get_path_list(df, "hifi_captain", "train")
            val_data_path_list += self.get_path_list(df, "hifi_captain", "val")
            test_data_path_list += self.get_path_list(df, "hifi_captain", "test")
            logger.info(
                f"""
                use hifi-captain
                {len(train_data_path_list)=}, {len(val_data_path_list)=}, {len(test_data_path_list)=}
                """
            )
        if self.cfg.data_choice.jvs.use:
            df = pd.read_csv(str(Path(self.cfg.path.jvs.df_path).expanduser()))
            df = df.loc[(df["data"] == "parallel100") | (df["data"] == "nonpara30")]
            train_data_path_list += self.get_path_list(df, "jvs", "train")
            val_data_path_list += self.get_path_list(df, "jvs", "val")
            test_data_path_list += self.get_path_list(df, "jvs", "test")
            logger.info(
                f"""
                use jvs
                {len(train_data_path_list)=}, {len(val_data_path_list)=}, {len(test_data_path_list)=}
                """
            )

        if stage == "fit":
            self.train_dataset = SpeechMemoryDataset(
                cfg=self.cfg,
                data_path_list=train_data_path_list,
                transform=SpeechMemoryTransform(self.cfg, "train"),
            )
            self.val_dataset = SpeechMemoryDataset(
                cfg=self.cfg,
                data_path_list=val_data_path_list,
                transform=SpeechMemoryTransform(self.cfg, "val"),
            )
        if stage == "test":
            self.test_dataset = SpeechMemoryDataset(
                cfg=self.cfg,
                data_path_list=test_data_path_list,
                transform=SpeechMemoryTransform(self.cfg, "test"),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=functools.partial(adjust_seq_lengths, cfg=self.cfg),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
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
