import subprocess


def main():
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_base_hubert_conv_decoder.py",
            "--seed",
            "42",
        ]
    )
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/run/exp_base_hubert_hubert_decoder.py",
            "--seed",
            "42",
        ]
    )


if __name__ == "__main__":
    main()
