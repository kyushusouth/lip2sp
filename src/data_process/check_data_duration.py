from pathlib import Path

import hydra
import librosa
import omegaconf
import polars as pl
import torchvision
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    audio_dir = Path(cfg.path.kablab.audio_dir).expanduser()
    video_dir = Path(cfg.path.kablab.video_dir).expanduser()
    df_path = Path(cfg.path.kablab.df_path).expanduser()
    df = pl.read_csv(str(df_path))
    results = []

    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        audio_path = audio_dir / row["speaker"] / f"{row['filename']}.wav"
        video_path = video_dir / row["speaker"] / f"{row['filename']}.mp4"
        if not audio_path.exists() or not video_path.exists():
            continue

        wav, _ = librosa.load(str(audio_path), sr=cfg.data.audio.sr)
        lip, _, _ = torchvision.io.read_video(
            str(video_path), pts_unit="sec", output_format="TCHW"
        )  # (T, C, H, W)

        wav_duration = wav.shape[0] / cfg.data.audio.sr
        lip_duration = lip.shape[0] / cfg.data.video.fps
        results.append(
            [
                row["speaker"],
                row["filename"],
                wav_duration,
                lip_duration,
            ]
        )

    df_results = pl.DataFrame(
        results, schema=["speaker", "filename", "dur_a", "dur_v"], orient="row"
    )
    df_results.write_csv("./check_data_duration_result.csv")


if __name__ == "__main__":
    main()
