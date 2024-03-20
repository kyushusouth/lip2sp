import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import wandb
from librosa.display import specshow


def save_mel(
    cfg: omegaconf.DictConfig, gt: np.ndarray, pred: np.ndarray, filename: str
) -> None:
    """
    gt, pred: (C, T)
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    specshow(
        data=gt,
        x_axis="time",
        y_axis="mel",
        sr=cfg["data"]["audio"]["sr"],
        hop_length=cfg["data"]["audio"]["hop_length"],
        fmin=cfg["data"]["audio"]["f_min"],
        fmax=cfg["data"]["audio"]["f_max"],
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Ground Truth")

    plt.subplot(2, 1, 2)
    specshow(
        data=pred,
        x_axis="time",
        y_axis="mel",
        sr=cfg["data"]["audio"]["sr"],
        hop_length=cfg["data"]["audio"]["hop_length"],
        fmin=cfg["data"]["audio"]["f_min"],
        fmax=cfg["data"]["audio"]["f_max"],
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Prediction")

    plt.tight_layout()

    with tempfile.TemporaryDirectory() as d:
        save_path = os.path.join(d, f"{filename}.png")
        plt.savefig(save_path)
        wandb.log({filename: wandb.Image(save_path)})

    plt.close()


def save_wav(
    cfg: omegaconf.DictConfig, gt: np.ndarray, pred: np.ndarray, filename: str
) -> None:
    """
    gt, pred: (T,)
    """
    gt /= np.max(np.abs(gt))
    pred /= np.max(np.abs(pred))
    wandb.log(
        {f"{filename}_gt": wandb.Audio(gt, sample_rate=cfg["data"]["audio"]["sr"])}
    )
    wandb.log(
        {f"{filename}_pred": wandb.Audio(pred, sample_rate=cfg["data"]["audio"]["sr"])}
    )
