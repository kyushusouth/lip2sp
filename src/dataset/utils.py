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


def get_token_index_mapping(cfg):
    phonemes = [
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
        "kw",
        "pau",
        "sil",
    ]

    if cfg.training.token_index.for_tts:
        extra_symbols = [
            "^",  # 文の先頭を表す特殊記号 <SOS>
            "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
            "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
            "_",  # ポーズ
            "#",  # アクセント句境界
            "[",  # ピッチの上がり位置
            "]",  # ピッチの下がり位置
        ]
    else:
        extra_symbols = [
            "^",  # 文の先頭を表す特殊記号 <SOS>
            "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
        ]

    _pad = "~"
    mask = "mask"
    token_list = [_pad] + extra_symbols + phonemes + [mask]
    token_to_id = {s: i for i, s in enumerate(token_list)}
    id_to_token = {i: s for i, s in enumerate(token_list)}
    return token_to_id, id_to_token
