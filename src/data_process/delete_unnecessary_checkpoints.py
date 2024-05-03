import shutil
from pathlib import Path


def main():
    necessary_checkpoint_list = [
        "/home/minami/lip2sp/checkpoints/base_hubert/20240425_120512",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240427_124850",
    ]
    checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    checkpoint_path_list = list(checkpoint_dir.glob("**/*.ckpt"))
    for checkpoint_path in checkpoint_path_list:
        if str(checkpoint_path.parent) in necessary_checkpoint_list:
            continue
        shutil.rmtree(str(checkpoint_path.parent))


if __name__ == "__main__":
    main()
