import pathlib
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import omegaconf
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from scipy.io.wavfile import write
from tqdm import tqdm

from src.main import start  # noqa: F401
from src.model.pwg import Generator
from src.pl_datamodule.base import BaseDataModule
from src.pl_module.base import LitBaseModel


def save_wav(
    cfg: omegaconf.DictConfig,
    wav: torch.Tensor,
    n_sample: int,
    save_path: pathlib.Path,
) -> None:
    wav = wav.squeeze(0).squeeze(0)
    wav = wav.cpu().detach().numpy()
    wav /= np.max(np.abs(wav))
    wav = wav.astype(np.float32)
    wav = wav[:n_sample]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(save_path), rate=cfg["data"]["audio"]["sr"], data=wav)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    checkpoint_save_dir = (
        Path(cfg["training"]["params"]["checkpoints_save_dir"]).expanduser()
        / start.CURRENT_TIME
    )
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["params"]["checkpoints_save_dir"] = str(checkpoint_save_dir)

    results_save_dir = (
        Path(cfg["training"]["params"]["results_save_dir"]).expanduser()
        / start.CURRENT_TIME
    )
    results_save_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["params"]["results_save_dir"] = str(results_save_dir)

    datamodule = BaseDataModule(cfg)
    model = LitBaseModel(cfg)

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
        accumulate_grad_batches=cfg["training"]["params"]["accumulate_grad_batches"],
        gradient_clip_val=cfg["training"]["params"]["gradient_clip_val"],
        gradient_clip_algorithm=cfg["training"]["params"]["gradient_clip_algorithm"],
    )
    # trainer.fit(model=model, datamodule=datamodule)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    model = LitBaseModel.load_from_checkpoint(
        cfg=cfg,
        checkpoint_path="/home/minami/lip2sp/checkpoints/20240317_011440/epoch=2-step=15-val_loss=0.775.ckpt",
        strict=True,
    )
    pwg = Generator(cfg)
    pretrained_dict = torch.load(cfg["model"]["pwg"]["model_path"])
    pwg.load_state_dict(pretrained_dict["gen"], strict=True)
    model = model.to(device)
    pwg = pwg.to(device)

    for batch in tqdm(test_dataloader):
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
        wav = wav.to(device)
        lip = lip.to(device)
        feature = feature.to(device)
        spk_emb = spk_emb.to(device)
        lip_len = lip_len.to(device)
        pred = model(
            lip=lip,
            audio=None,
            lip_len=lip_len,
            spk_emb=spk_emb,
        )
        noise = torch.randn(
            pred.shape[0], 1, pred.shape[-1] * cfg["data"]["audio"]["hop_length"]
        ).to(device=pred.device, dtype=pred.dtype)
        wav_pred = pwg(noise, pred)
        wav_abs = pwg(noise, feature)
        n_sample_min = min(wav.shape[-1], wav_pred.shape[-1], wav_abs.shape[-1])

        save_wav(
            cfg=cfg,
            wav=wav,
            n_sample=n_sample_min,
            save_path=(results_save_dir / speaker[0] / filename[0] / "gt.wav"),
        )
        save_wav(
            cfg=cfg,
            wav=wav_abs,
            n_sample=n_sample_min,
            save_path=(results_save_dir / speaker[0] / filename[0] / "abs.wav"),
        )
        save_wav(
            cfg=cfg,
            wav=wav_pred,
            n_sample=n_sample_min,
            save_path=(results_save_dir / speaker[0] / filename[0] / "pred.wav"),
        )


if __name__ == "__main__":
    main()
