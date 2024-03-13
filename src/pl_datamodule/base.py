import lightning as L
import omegaconf
from torch.utils.data import DataLoader


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = None
            self.val_dataset = None
        if stage == "test":
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()
