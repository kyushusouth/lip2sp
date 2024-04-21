from pathlib import Path

import hydra
import lightning as L
import omegaconf
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from main import on_start  # noqa: F401
from src.pl_datamodule.base import BaseDataModule
from src.pl_module.pwg import LitPWG


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    cfg["training"]["checkpoints_save_dir"] = str(
        Path(cfg["training"]["checkpoints_save_dir"]).expanduser()
        / on_start.CURRENT_TIME
    )

    datamodule = BaseDataModule(cfg)

    if cfg["training"]["finetune"]:
        model = LitPWG.load_from_checkpoint(
            checkpoint_path=cfg["training"]["finetune_start_model_path"],
            cfg=cfg,
        )
    else:
        model = LitPWG(cfg)

    wandb_logger = WandbLogger(
        project=cfg["training"]["wandb"]["project_name"],
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
        name=cfg["training"]["wandb"]["run_name"],
        group=cfg["training"]["wandb"]["group_name"],
    )

    wandb_logger.watch(model, log="all", log_graph=True, log_freq=10)

    trainer = L.Trainer(
        default_root_dir=cfg["training"]["wandb"]["default_root_dir"],
        max_epochs=cfg["training"]["max_epoch"],
        callbacks=[
            EarlyStopping(
                monitor=cfg["training"]["monitoring_metric"],
                mode=cfg["training"]["monitoring_mode"],
                patience=cfg["training"]["early_stopping_patience"],
            ),
            ModelCheckpoint(
                monitor=cfg["training"]["monitoring_metric"],
                mode=cfg["training"]["monitoring_mode"],
                every_n_epochs=cfg["training"][
                    "save_checkpoint_every_n_epochs"
                ],
                save_top_k=cfg["training"]["save_checkpoint_top_k"],
                dirpath=cfg["training"]["checkpoints_save_dir"],
                filename="{epoch}-{step}-{val_loss:.3f}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        limit_train_batches=float(cfg["training"]["limit_train_batches"]),
        limit_val_batches=float(cfg["training"]["limit_val_batches"]),
        limit_test_batches=float(cfg["training"]["limit_test_batches"]),
        profiler=None,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        precision=cfg["training"]["precision"],
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # trainer.test(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path="best",
    # )

    wandb.finish()


if __name__ == "__main__":
    main()
