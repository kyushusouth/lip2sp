from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import pickle
import numpy as np
import hydra
import omegaconf


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    phone_lst = [
        "a",
        "aː",
        "b",
        "bʲ",
        "bʲː",
        "bː",
        "c",
        "cː",
        "d",
        "dz",
        "dzː",
        "dʑ",
        "dʑː",
        "dʲ",
        "dʲː",
        "dː",
        "e",
        "eː",
        "h",
        "hː",
        "i",
        "iː",
        "i̥",
        "j",
        "k",
        "kː",
        "m",
        "mʲ",
        "mʲː",
        "mː",
        "n",
        "nː",
        "o",
        "oː",
        "p",
        "pʲ",
        "pʲː",
        "pː",
        "s",
        "sː",
        "t",
        "ts",
        "tsː",
        "tɕ",
        "tɕː",
        "tʲ",
        "tʲː",
        "tː",
        "v",
        "vʲ",
        "w",
        "wː",
        "z",
        "ç",
        "çː",
        "ŋ",
        "ɕ",
        "ɕː",
        "ɟ",
        "ɟː",
        "ɡ",
        "ɡː",
        "ɨ",
        "ɨː",
        "ɨ̥",
        "ɯ",
        "ɯː",
        "ɯ̥",
        "ɰ̃",
        "ɲ",
        "ɲː",
        "ɴ",
        "ɴː",
        "ɸ",
        "ɸʲ",
        "ɸʲː",
        "ɸː",
        "ɾ",
        "ɾʲ",
        "ɾʲː",
        "ɾː",
        "ʑ",
        "ʔ",
    ]
    phone_lst = np.array(phone_lst, dtype="U100").reshape(-1, 1)

    onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    onehot_encoder.fit(phone_lst)

    save_path = Path(cfg.path.kablab.onehot_encoder_dir).expanduser() / "phones.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(save_path), "wb") as f:
        pickle.dump(onehot_encoder, f)


if __name__ == "__main__":
    main()
