import hydra
import omegaconf

from src import start  # noqa: F401
from src.pl_datamodule.base import BaseDataModule


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    datamodule = BaseDataModule(cfg)
    datamodule.setup("fit")


if __name__ == "__main__":
    main()
