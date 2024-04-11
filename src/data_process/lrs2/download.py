import subprocess
from pathlib import Path


def main():
    url_list = [
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partaa",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partab",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partac",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partad",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partae",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/pretrain.txt",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/train.txt",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/val.txt",
        "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/test.txt",
    ]
    lrs2_dir = Path("~/dataset/lrs2").expanduser()
    lrs2_dir.mkdir(parents=True, exist_ok=True)
    for url in url_list:
        filename = url.split("/")[-1]
        subprocess.run(
            [
                "curl",
                "--output",
                str(lrs2_dir / filename),
                "--user",
                "lrs753:2U3Vv2Hy",
                url,
            ]
        )


if __name__ == "__main__":
    main()
