from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm


def check_hifi():
    audio_dir_hifi = Path("~/dataset/hi-fi-captain").expanduser()
    audio_dir_hifi_cut = Path("~/dataset/hi-fi-captain_cut_silence").expanduser()
    audio_path_list = list(audio_dir_hifi.glob("**/*.wav"))

    diff_list = []
    for audio_path in tqdm(audio_path_list):
        audio_path_cut = Path(
            str(audio_path).replace(str(audio_dir_hifi), str(audio_dir_hifi_cut))
        )

        if not audio_path.exists() or not audio_path_cut.exists():
            continue

        wav, _ = librosa.load(str(audio_path))
        wav_cut, _ = librosa.load(str(audio_path_cut))
        diff = abs(wav.shape[0] - wav_cut.shape[0])
        diff_list.append([diff, audio_path, audio_path_cut])

    df = pd.DataFrame(
        data={
            "diff": [diff_list[i][0] for i in range(len(diff_list))],
            "audio_path": [diff_list[i][1] for i in range(len(diff_list))],
            "audio_path_cut": [diff_list[i][2] for i in range(len(diff_list))],
        }
    )
    df = df.sort_values(["diff"])
    df.to_csv("./audio_diff_hifi.csv", index=False)


def check_jvs():
    audio_dir_jvs = Path("~/dataset/jvs_ver1").expanduser()
    audio_dir_jvs_cut = Path("~/dataset/jvs_ver1_cut_silence").expanduser()
    audio_path_list = list(audio_dir_jvs.glob("**/*.wav"))

    diff_list = []
    for audio_path in tqdm(audio_path_list):
        audio_path_cut = Path(
            str(audio_path).replace(str(audio_dir_jvs), str(audio_dir_jvs_cut))
        )

        if not audio_path.exists() or not audio_path_cut.exists():
            continue

        wav, _ = librosa.load(str(audio_path))
        wav_cut, _ = librosa.load(str(audio_path_cut))
        diff = abs(wav.shape[0] - wav_cut.shape[0])
        diff_list.append([diff, audio_path, audio_path_cut])

    df = pd.DataFrame(
        data={
            "diff": [diff_list[i][0] for i in range(len(diff_list))],
            "audio_path": [diff_list[i][1] for i in range(len(diff_list))],
            "audio_path_cut": [diff_list[i][2] for i in range(len(diff_list))],
        }
    )
    df = df.sort_values(["diff"])
    df.to_csv("./audio_diff_jvs.csv", index=False)


def main():
    # check_hifi()
    check_jvs()


if __name__ == "__main__":
    main()
