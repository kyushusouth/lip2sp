import subprocess


def main():
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_hifigan.py",
        ]
    )
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_base_hubert_hubert_decoder.py",
        ]
    )


if __name__ == "__main__":
    main()
