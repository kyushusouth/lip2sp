"""
音声における無音区間の除去を検討。
しかし、動画と音声が実際若干ずれている。動画が遅れ気味。
元データの時点で前処理がうまくいってないと予想されるが、今からそこに立ち返って修正するのは厳しい。
完全にアラインメントがとれていなくても、時系列全体を考慮する処理が入っているので学習はできていると仮定するしかない。
"""

import math
import shutil
import subprocess
from pathlib import Path

import hydra
import omegaconf
import pydub
import pydub.silence
import torchvision
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    audio_dir = Path(cfg.path.kablab.audio_dir).expanduser()
    video_dir = Path(cfg.path.kablab.video_dir).expanduser()
    audio_cut_silence_dir = Path(cfg.path.kablab.audio_cut_silence_dir).expanduser()
    video_cut_silence_dir = Path(cfg.path.kablab.video_cut_silence_dir).expanduser()
    merged_dir = Path("/home/minami/dataset/lip/video_audio_merged")
    if audio_cut_silence_dir.exists():
        shutil.rmtree(audio_cut_silence_dir)
    if video_cut_silence_dir.exists():
        shutil.rmtree(video_cut_silence_dir)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    audio_src_path_lst = list(audio_dir.glob("**/*.wav"))

    for audio_src_path in tqdm(audio_src_path_lst):
        video_src_path = (
            video_dir / audio_src_path.parents[0].name / f"{audio_src_path.stem}.mp4"
        )
        if (
            video_src_path.parents[0].name == "M04_kablab"
            or video_src_path.parents[0].name == "F01_kablab"
        ):
            continue

        if not audio_src_path.exists() or not video_src_path.exists():
            continue

        audio_dest_path = (
            audio_cut_silence_dir / audio_src_path.parents[0].name / audio_src_path.name
        )
        audio_dest_path.parent.mkdir(parents=True, exist_ok=True)
        video_dest_path = (
            video_cut_silence_dir / video_src_path.parents[0].name / video_src_path.name
        )
        video_dest_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / video_src_path.parents[0].name / video_src_path.name
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        sound = pydub.AudioSegment.from_file(str(audio_src_path))
        silent_ranges = pydub.silence.detect_silence(
            sound, min_silence_len=500, silence_thresh=-40
        )

        lip, _, _ = torchvision.io.read_video(
            str(video_src_path), pts_unit="sec", output_format="THWC"
        )  # (T, H, W, C)

        # 無音区間の除去を行うが、動画との兼ね合いもあるので最後の無音だけ除去対象にする
        if len(silent_ranges) == 0 or not math.isclose(
            silent_ranges[-1][1], sound.duration_seconds * 1000, abs_tol=100
        ):
            sound.export(
                str(audio_dest_path),
                format="wav",
                parameters=["-ar", f"{sound.frame_rate}"],
            )
            torchvision.io.write_video(
                filename=str(video_dest_path),
                video_array=lip,
                fps=cfg.data.video.fps,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_dest_path),
                    "-i",
                    str(audio_dest_path),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    str(merged_path),
                ]
            )
        else:
            no_silence_audio = sound[: silent_ranges[-1][0] + 200]
            if sound.frame_rate != no_silence_audio.frame_rate:
                raise ValueError()

            # 40ms / frame
            lip_trimmed = lip[: (silent_ranges[-1][0] + 200) // 40]

            no_silence_audio.export(
                str(audio_dest_path),
                format="wav",
                parameters=["-ar", f"{sound.frame_rate}"],
            )
            torchvision.io.write_video(
                filename=str(video_dest_path),
                video_array=lip_trimmed,
                fps=cfg.data.video.fps,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_dest_path),
                    "-i",
                    str(audio_dest_path),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    str(merged_path),
                ]
            )


if __name__ == "__main__":
    main()
