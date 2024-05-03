from pathlib import Path

import omegaconf
import torch

from src.model.components.avhubert import AVHuBERT
from src.model.components.raven import RAVEn
from src.model.components.vatlm import VATLM


def load_avhubert(cfg: omegaconf.DictConfig):
    avhubert = AVHuBERT(cfg.model.avhubert)
    ckpt_path = Path(cfg.model.avhubert.ckpt_path).expanduser()
    if cfg.model.avhubert.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))["avhubert"]
        avhubert.load_state_dict(pretrained_dict, strict=True)
    return avhubert


def load_raven(cfg: omegaconf.DictConfig):
    raven = RAVEn(cfg.model.raven).encoder
    ckpt_path = Path(cfg.model.raven.ckpt_path).expanduser()
    if cfg.model.raven.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))
        raven.load_state_dict(pretrained_dict, strict=True)
    return raven


def load_vatlm(cfg: omegaconf.DictConfig):
    vatlm = VATLM(
        cfg=cfg.model.vatlm.cfg,
        task_cfg=cfg.model.vatlm.task_cfg,
        dictionaries=None,
    )
    ckpt_path = Path(cfg.model.vatlm.ckpt_path).expanduser()
    if cfg.model.vatlm.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))["vatlm"]
        vatlm.load_state_dict(pretrained_dict, strict=True)
    return vatlm
