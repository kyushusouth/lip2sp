"""
montreal forced alignerによるforced alignmentを行う場合、condaの仮想環境をアクティベートする。

mfa align --overwrite --clean --use_mp /home/minami/dataset/lip/forced_alignment_results/data japanese_mfa japanese_mfa /home/minami/dataset/lip/forced_alignment_results/results
"""

import shutil
from pathlib import Path

import hydra
import omegaconf
import polars as pl
from tqdm import tqdm


def kablab():
    text_dir = Path("/home/minami/dataset/lip/utt")
    audio_dir = Path("/home/minami/dataset/lip/wav")
    save_dir = Path("/home/minami/dataset/lip/forced_alignment_results/data")
    n_limit = 1000000

    speaker_list = [path.stem for path in audio_dir.glob("*")]
    for speaker in speaker_list:
        audio_path_list = list((audio_dir / speaker).glob("**/*.wav"))[:n_limit]
        for audio_path in tqdm(audio_path_list):
            text_path = text_dir / f"{audio_path.stem}.txt"
            save_dir_spk = save_dir / speaker
            save_dir_spk.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(audio_path), str(save_dir_spk / audio_path.name))
            shutil.copy(str(text_path), str(save_dir_spk / f"{text_path.stem}.lab"))


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    audio_dir = Path(cfg.path.jvs.audio_dir).expanduser()
    text_dir = Path(cfg.path.jvs.text_dir).expanduser()
    save_dir = Path(cfg.path.jvs.forced_alignment_results_dir).expanduser()
    df = pl.read_csv(str(Path(cfg.path.jvs.df_path).expanduser()))

    for row in df.iter_rows(named=True):
        utt = (
            pl.read_csv(
                str(text_dir / row["speaker"] / row["data"] / "transcripts_utf8.txt"),
                has_header=False,
                separator=":",
            )
            .rename({"column_1": "filename", "column_2": "utt"})
            .filter(pl.col("filename") == row["filename"])
            .select("utt")
            .to_numpy()[0][0]
        )


if __name__ == "__main__":
    main()
