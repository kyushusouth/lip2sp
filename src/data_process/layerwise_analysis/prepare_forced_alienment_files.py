"""
montreal forced alignerによるforced alignmentを行う場合、condaの仮想環境をアクティベートする。

mfa align --overwrite --clean --use_mp /home/minami/dataset/lip/forced_alignment_results/data japanese_mfa japanese_mfa /home/minami/dataset/lip/forced_alignment_results/results
"""

from pathlib import Path
import shutil
from tqdm import tqdm


def main():
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


if __name__ == "__main__":
    main()
