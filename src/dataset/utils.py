from pathlib import Path

import numpy as np
import omegaconf


def get_spk_emb(cfg: omegaconf.DictConfig):
    spk_emb_dict = {}
    emb_dir = Path(cfg.path.kablab.emb_dir).expanduser()
    data_path_list = emb_dir.glob("**/*.npy")
    for data_path in data_path_list:
        speaker = data_path.parents[0].name
        data_path = emb_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def get_spk_emb_jvs(cfg: omegaconf.DictConfig):
    data_path = Path(cfg.path.jvs.emb_dir).expanduser()
    speaker_list = [f"jvs{i:03d}" for i in range(1, 101)]
    spk_emb_dict = {}
    for speaker in speaker_list:
        emb = np.load(str(data_path / speaker / "emb.npy"))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def get_spk_emb_hifi_captain(cfg: omegaconf.DictConfig):
    spk_emb_dict = {}
    data_dir = Path(cfg.path.hifi_captain.emb_dir).expanduser()
    for speaker in ["female", "male"]:
        data_path = data_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def get_spk_emb_jsut(cfg: omegaconf.DictConfig):
    spk_emb_dict = {}
    data_dir = Path(cfg.path.jsut.emb_dir).expanduser()
    for speaker in ["female"]:
        data_path = data_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict
