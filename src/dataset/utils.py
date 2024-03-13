from pathlib import Path

import numpy as np


def get_spk_emb(cfg):
    spk_emb_dict = {}
    emb_dir = Path(cfg['path']['kablab']['emb_dir']).expanduser()
    data_path_list = emb_dir.glob("**/*.npy")
    for data_path in data_path_list:
        speaker = data_path.parents[0].name
        data_path = emb_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict
