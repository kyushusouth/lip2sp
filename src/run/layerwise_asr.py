import subprocess


def main():
    for model_name in ["rinna/japanese-hubert-base", "rinna/japanese-wav2vec2-base"]:
        for layer_index in range(1, 13):
            subprocess.run(
                [
                    "python",
                    "/home/minami/lip2sp/src/main/layerwise_asr.py",
                    f"model.layerwise_asr.layer_index={layer_index}",
                    f"model.layerwise_asr.model_name={model_name}",
                ]
            )


if __name__ == "__main__":
    main()
