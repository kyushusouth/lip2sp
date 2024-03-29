from pathlib import Path

import hydra
import lightning as L
import omegaconf
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.main import start  # noqa: F401
from src.pl_datamodule.base import BaseDataModule
from src.pl_module.pwg import LitPWG


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    checkpoint_save_dir = (
        Path(cfg["training"]["params"]["checkpoints_save_dir"]).expanduser()
        / start.CURRENT_TIME
    )
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["params"]["checkpoints_save_dir"] = str(checkpoint_save_dir)

    datamodule = BaseDataModule(cfg)
    model = LitPWG(cfg)

    wandb_logger = WandbLogger(
        project=cfg["training"]["params"]["wandb"]["project_name"],
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
        name=cfg["training"]["params"]["wandb"]["run_name"],
        group=cfg["training"]["params"]["wandb"]["group_name"],
    )

    wandb_logger.watch(model, log="all", log_graph=True, log_freq=10)

    trainer = L.Trainer(
        default_root_dir=cfg["training"]["params"]["wandb"]["default_root_dir"],
        max_epochs=cfg["training"]["params"]["max_epoch"],
        callbacks=[
            EarlyStopping(
                monitor=cfg["training"]["params"]["monitoring_metric"],
                mode=cfg["training"]["params"]["monitoring_mode"],
                patience=cfg["training"]["params"]["early_stopping_patience"],
            ),
            ModelCheckpoint(
                monitor=cfg["training"]["params"]["monitoring_metric"],
                mode=cfg["training"]["params"]["monitoring_mode"],
                every_n_epochs=cfg["training"]["params"][
                    "save_checkpoint_every_n_epochs"
                ],
                save_top_k=cfg["training"]["params"]["save_checkpoint_top_k"],
                dirpath=str(checkpoint_save_dir),
                filename="{epoch}-{step}-{val_loss:.3f}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        limit_train_batches=float(cfg["training"]["params"]["limit_train_batches"]),
        limit_val_batches=float(cfg["training"]["params"]["limit_val_batches"]),
        limit_test_batches=float(cfg["training"]["params"]["limit_test_batches"]),
        profiler=None,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=cfg["training"]["params"]["log_every_n_steps"],
        precision=cfg["training"]["params"]["precision"],
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
