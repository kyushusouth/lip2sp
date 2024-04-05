from pathlib import Path

import hydra
import omegaconf
import pandas as pd
import pydub
from tqdm import tqdm


def process(src_path: Path, dest_path: Path) -> None:
    sound = pydub.AudioSegment.from_file(str(src_path))
    chunks = pydub.silence.split_on_silence(
        sound, min_silence_len=500, silence_thresh=-40, keep_silence=100
    )
    no_silence_audio = pydub.AudioSegment.empty()
    for chunk in chunks:
        no_silence_audio += chunk
    if sound.frame_rate != no_silence_audio.frame_rate:
        raise ValueError()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    no_silence_audio.export(
        str(dest_path), format="wav", parameters=["-ar", f"{sound.frame_rate}"]
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    df_hifi = pd.read_csv(
        str(Path(cfg["path"]["hifi_captain"]["df_path"]).expanduser())
    )
    src_dir = Path(cfg["path"]["hifi_captain"]["data_dir"]).expanduser()
    dest_dir = Path(cfg["path"]["hifi_captain"]["data_dir_cut_silence"]).expanduser()
    for i, row in tqdm(df_hifi.iterrows(), total=len(df_hifi)):
        src_path = (
            src_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.wav'
        )
        dest_path = (
            dest_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.wav'
        )
        try:
            process(src_path, dest_path)
        except:
            continue

    df_jvs = pd.read_csv(str(Path(cfg["path"]["jvs"]["df_path"]).expanduser()))
    src_dir = Path(cfg["path"]["jvs"]["data_dir"]).expanduser()
    dest_dir = Path(cfg["path"]["jvs"]["data_dir_cut_silence"]).expanduser()
    for i, row in tqdm(df_jvs.iterrows(), total=len(df_jvs)):
        src_path = (
            src_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.wav'
        )
        dest_path = (
            dest_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.wav'
        )
        try:
            process(src_path, dest_path)
        except:
            continue


if __name__ == "__main__":
    main()
