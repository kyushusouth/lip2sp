import subprocess


def main():
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_baseline_hifigan_finetuning.py",
        ]
    )
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_base_hubert_conv_decoder.py",
        ]
    )


if __name__ == "__main__":
    main()
