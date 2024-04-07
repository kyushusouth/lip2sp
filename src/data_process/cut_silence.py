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


def process_hifi_captain(cfg: omegaconf.DictConfig) -> None:
    src_dir = Path(cfg["path"]["hifi_captain"]["data_dir"]).expanduser()
    dest_dir = Path(str(src_dir).replace(src_dir.name, f"{src_dir.name}_cut_silence"))
    df_src_path = Path(cfg["path"]["hifi_captain"]["df_path"]).expanduser()
    df_dest_path = Path(
        str(df_src_path).replace(
            df_src_path.parent.name, f"{df_src_path.parent.name}_cut_silence"
        )
    )
    df_src = pd.read_csv(str(df_src_path))

    for i, row in tqdm(df_src.iterrows(), total=len(df_src)):
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

    df_src.to_csv(str(df_dest_path), index=False)


def process_jvs(cfg: omegaconf.DictConfig) -> None:
    src_dir = Path(cfg["path"]["jsut"]["data_dir"]).expanduser()
    dest_dir = Path(str(src_dir).replace(src_dir.name, f"{src_dir.name}_cut_silence"))
    df_src_path = Path(cfg["path"]["jsut"]["df_path"]).expanduser()
    df_dest_path = Path(
        str(df_src_path).replace(
            df_src_path.parent.name, f"{df_src_path.parent.name}_cut_silence"
        )
    )
    df_src = pd.read_csv(str(df_src_path))

    for i, row in tqdm(df_src.iterrows(), total=len(df_src)):
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

    df_src.to_csv(str(df_dest_path), index=False)


def process_jsut(cfg: omegaconf.DictConfig) -> None:
    src_dir = Path(cfg["path"]["jsut"]["data_dir"]).expanduser()
    dest_dir = Path(str(src_dir).replace(src_dir.name, f"{src_dir.name}_cut_silence"))
    df_src_path = Path(cfg["path"]["jsut"]["df_path"]).expanduser()
    df_dest_path = Path(
        str(df_src_path).replace(
            df_src_path.parent.name, f"{df_src_path.parent.name}_cut_silence"
        )
    )
    df_src = pd.read_csv(str(df_src_path))

    for i, row in tqdm(df_src.iterrows(), total=len(df_src)):
        src_path = src_dir / row["dirname"] / "wav" / f'{row["filename"]}.wav'
        dest_path = dest_dir / row["dirname"] / "wav" / f'{row["filename"]}.wav'
        try:
            process(src_path, dest_path)
        except:
            continue

    df_src.to_csv(str(df_dest_path), index=False)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    # process_hifi_captain(cfg)
    # process_jvs(cfg)
    process_jsut(cfg)


if __name__ == "__main__":
    main()
