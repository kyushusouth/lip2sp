import os
import shutil
from pathlib import Path

"""
hifigan
"/home/minami/lip2sp/checkpoints/hifigan/20240412_173956"
"/home/minami/lip2sp/checkpoints/hifigan/20240413_073833"
"/home/minami/lip2sp/checkpoints/hifigan/20240412_173956"
"/home/minami/lip2sp/checkpoints/hifigan/20240413_073833"
"/home/minami/lip2sp/checkpoints/hifigan/20240414_150602"
"/home/minami/lip2sp/checkpoints/hifigan/20240415_082235"
"/home/minami/lip2sp/checkpoints/hifigan/20240415_145735"
"/home/minami/lip2sp/checkpoints/hifigan/20240416_050038"
"/home/minami/lip2sp/checkpoints/hifigan/20240416_100708"
"/home/minami/lip2sp/checkpoints/hifigan/20240417_032839"
"/home/minami/lip2sp/checkpoints/hifigan/20240419_141114"
"/home/minami/lip2sp/checkpoints/hifigan/20240420_041048"
"/home/minami/lip2sp/checkpoints/hifigan/20240419_141114"
"/home/minami/lip2sp/checkpoints/hifigan/20240420_041048"
"/home/minami/lip2sp/checkpoints/hifigan/20240420_094814"
"/home/minami/lip2sp/checkpoints/hifigan/20240421_030614"
"/home/minami/lip2sp/checkpoints/hifigan/20240420_094814"
"/home/minami/lip2sp/checkpoints/hifigan/20240421_030614"
"/home/minami/lip2sp/checkpoints/hifigan/20240417_183535"
"/home/minami/lip2sp/checkpoints/hifigan/20240418_083428"
"/home/minami/lip2sp/checkpoints/hifigan/20240418_141038"
"/home/minami/lip2sp/checkpoints/hifigan/20240419_072830"

base_hubert
"/home/minami/lip2sp/checkpoints/base_hubert/20240413_120846"
"/home/minami/lip2sp/checkpoints/base_hubert/20240414_130415"
"/home/minami/lip2sp/checkpoints/base_hubert/20240415_135403"
"/home/minami/lip2sp/checkpoints/base_hubert/20240416_093302"
"/home/minami/lip2sp/checkpoints/base_hubert/20240417_090252"
"/home/minami/lip2sp/checkpoints/base_hubert/20240420_084323"
"/home/minami/lip2sp/checkpoints/base_hubert/20240420_084323"
"/home/minami/lip2sp/checkpoints/base_hubert/20240420_092950"
"/home/minami/lip2sp/checkpoints/base_hubert/20240421_083903"
"/home/minami/lip2sp/checkpoints/base_hubert/20240421_083903"
"/home/minami/lip2sp/checkpoints/base_hubert/20240421_092638"
"/home/minami/lip2sp/checkpoints/base_hubert/20240418_130654"
"/home/minami/lip2sp/checkpoints/base_hubert/20240418_135209"
"/home/minami/lip2sp/checkpoints/base_hubert/20240419_130244"
"/home/minami/lip2sp/checkpoints/base_hubert/20240419_135251"
"""


def delete_files(checkpoint_dir):
    necessary_dir_list = [
        "/home/minami/lip2sp/checkpoints/hifigan/20240412_173956",
        "/home/minami/lip2sp/checkpoints/hifigan/20240413_073833",
        "/home/minami/lip2sp/checkpoints/hifigan/20240412_173956",
        "/home/minami/lip2sp/checkpoints/hifigan/20240413_073833",
        "/home/minami/lip2sp/checkpoints/hifigan/20240414_150602",
        "/home/minami/lip2sp/checkpoints/hifigan/20240415_082235",
        "/home/minami/lip2sp/checkpoints/hifigan/20240415_145735",
        "/home/minami/lip2sp/checkpoints/hifigan/20240416_050038",
        "/home/minami/lip2sp/checkpoints/hifigan/20240416_100708",
        "/home/minami/lip2sp/checkpoints/hifigan/20240417_032839",
        "/home/minami/lip2sp/checkpoints/hifigan/20240419_141114",
        "/home/minami/lip2sp/checkpoints/hifigan/20240420_041048",
        "/home/minami/lip2sp/checkpoints/hifigan/20240419_141114",
        "/home/minami/lip2sp/checkpoints/hifigan/20240420_041048",
        "/home/minami/lip2sp/checkpoints/hifigan/20240420_094814",
        "/home/minami/lip2sp/checkpoints/hifigan/20240421_030614",
        "/home/minami/lip2sp/checkpoints/hifigan/20240420_094814",
        "/home/minami/lip2sp/checkpoints/hifigan/20240421_030614",
        "/home/minami/lip2sp/checkpoints/hifigan/20240417_183535",
        "/home/minami/lip2sp/checkpoints/hifigan/20240418_083428",
        "/home/minami/lip2sp/checkpoints/hifigan/20240418_141038",
        "/home/minami/lip2sp/checkpoints/hifigan/20240419_072830",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240413_120846",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240414_130415",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240415_135403",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240416_093302",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240417_090252",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240420_084323",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240420_084323",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240420_092950",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240421_083903",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240421_083903",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240421_092638",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240418_130654",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240418_135209",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240419_130244",
        "/home/minami/lip2sp/checkpoints/base_hubert/20240419_135251",
    ]
    necessary_dir_list = list(set(necessary_dir_list))
    checkpoint_dir_list = list(checkpoint_dir.glob("*"))
    for d in checkpoint_dir_list:
        if str(d) in necessary_dir_list:
            continue
        shutil.rmtree(str(d))


def main():
    checkpoint_dir_hifigan = Path("/home/minami/lip2sp/checkpoints/hifigan")
    checkpoint_dir_base_hubert = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    delete_files(checkpoint_dir_hifigan)
    delete_files(checkpoint_dir_base_hubert)


if __name__ == "__main__":
    main()
