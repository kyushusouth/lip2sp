import hydra
import lightning as L
import omegaconf
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src import start  # noqa: F401
from src.pl_datamodule.base import BaseDataModule
from src.pl_module.base import LitBaseModel


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    datamodule = BaseDataModule(cfg)
    model = LitBaseModel(cfg)
    wandb_logger = WandbLogger(
        project=cfg["training"]["params"]["wandb"]["project_name"],
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
        name=None,
        group=None,
    )
    wandb_logger.watch(model, log="all")
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
                dirpath=f"./logs/{start.CURRENT_TIME}",
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
        accumulate_grad_batches=cfg["training"]["params"]["accumulate_grad_batches"],
        gradient_clip_val=cfg["training"]["params"]["gradient_clip_val"],
        gradient_clip_algorithm=cfg["training"]["params"]["gradient_clip_algorithm"],
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
