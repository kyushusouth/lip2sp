import logging
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
from src.pl_module.base_hubert_2 import LitBaseHuBERT2Module

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    seed_everything(seed=cfg.training.seed)

    ckpt_path_lst = [
        # "/home/minami/lip2sp/checkpoints/base_hubert_2/20241007_003155/epoch:45-step:2300.ckpt",
        # "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_233941/epoch:45-step:2300.ckpt",
        "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_134056/epoch:32-step:1650.ckpt",
        "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_150112/epoch:18-step:950.ckpt",
        "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_155146/epoch:17-step:900.ckpt",
        "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_164631/epoch:6-step:350.ckpt",
    ]
    ckpt_path_lst_map = {
        "20241007_003155": "20241023_151902",
        "20241004_233941": "20241023_162612",
        "20241005_134056": "20241023_173201",
        "20241005_150112": "20241023_174120",
        "20241005_155146": "20241023_175041",
        "20241005_164631": "20241023_180001",
    }
    checkpoints_save_dir_orig = cfg.training.checkpoints_save_dir

    for ckpt_path in ckpt_path_lst:
        ckpt_path = Path(ckpt_path)

        cfg.training.checkpoints_save_dir = str(
            Path(checkpoints_save_dir_orig).expanduser()
            / ckpt_path_lst_map[ckpt_path.parents[0].name]
        )

        datamodule = BaseHuBERT2DataModule(cfg)

        if cfg.training.finetune:
            model = LitBaseHuBERT2Module.load_from_checkpoint(
                checkpoint_path=cfg.training.finetune_start_model_path, cfg=cfg
            )
        else:
            model = LitBaseHuBERT2Module(cfg=cfg)

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
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            gradient_clip_val=cfg.training.gradient_clip_val,
            gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
            num_sanity_val_steps=0,
        )

        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(ckpt_path),
        )

        wandb.finish()


if __name__ == "__main__":
    main()
