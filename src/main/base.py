import hydra
import omegaconf

from src import start  # noqa: F401
from src.pl_datamodule.base import BaseDataModule


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    datamodule = BaseDataModule(cfg)
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    for batch in train_dataloader:
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            filename,
        ) = batch
        print(wav.shape)
        print(lip.shape)
        print(feature.shape)
        print(feature_avhubert.shape)
        print(spk_emb.shape)
        print(feature_len)
        print(lip_len)
        print(speaker)
        print(filename)


if __name__ == "__main__":
    main()
