import datetime
import os

import dotenv
import hydra
import omegaconf
import wandb

from src import start
from src.pl_datamodule.base import BaseDataModule


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    datamodule = BaseDataModule(cfg)


if __name__ == "__main__":
    main()
