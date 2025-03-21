from pathlib import Path

import hydra
import lightning as L
import omegaconf
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.main import on_start  # noqa: F401
from src.main.on_end import rename_checkpoint_file
from src.pl_datamodule.base_hubert_2 import BaseHuBERT2DataModule
from src.pl_module.hifigan_base_hubert_2 import LitHiFiGANBaseHuBERT2Model


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    seed_everything(seed=cfg.training.seed)

    cfg.training.checkpoints_save_dir = str(
        Path(cfg.training.checkpoints_save_dir).expanduser() / on_start.CURRENT_TIME
    )

    datamodule = BaseHuBERT2DataModule(cfg)

    if cfg.training.finetune:
        model = LitHiFiGANBaseHuBERT2Model.load_from_checkpoint(
            checkpoint_path=cfg.training.finetune_start_model_path,
            cfg=cfg,
        )
    else:
        model = LitHiFiGANBaseHuBERT2Model(cfg)

    wandb_logger = WandbLogger(
        project=cfg.training.wandb.project_name,
        log_model=False,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
        name=cfg.training.wandb.run_name,
        group=cfg.training.wandb.group_name,
    )

    wandb_logger.watch(model, log="all", log_graph=True, log_freq=10)

    trainer = L.Trainer(
        default_root_dir=cfg.training.wandb.default_root_dir,
        max_epochs=cfg.training.max_epoch,
        callbacks=[
            EarlyStopping(
                monitor=cfg.training.monitoring_metric,
                mode=cfg.training.monitoring_mode,
                patience=cfg.training.early_stopping_patience,
            ),
            ModelCheckpoint(
                monitor=cfg.training.monitoring_metric,
                mode=cfg.training.monitoring_mode,
                every_n_epochs=cfg.training.save_checkpoint_every_n_epochs,
                save_top_k=cfg.training.save_checkpoint_top_k,
                dirpath=cfg.training.checkpoints_save_dir,
                filename="{epoch}-{step}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        limit_train_batches=float(cfg.training.limit_train_batches),
        limit_val_batches=float(cfg.training.limit_val_batches),
        limit_test_batches=float(cfg.training.limit_test_batches),
        profiler=None,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=cfg.training.log_every_n_steps,
        precision=cfg.training.precision,
        num_sanity_val_steps=0,
    )

    if cfg.training.resume:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.training.resume_checkpoint_path,
        )
    else:
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

    rename_checkpoint_file(cfg.training.checkpoints_save_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
