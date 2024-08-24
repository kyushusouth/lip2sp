import pickle
from pathlib import Path

import hydra
import librosa
import numpy as np
import omegaconf
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModel


def process_save_numerical_features(
    cfg: omegaconf.DictConfig,
    audio_path: Path,
    hubert_feature_prj_output_path: Path,
    hubert_encoder_output_path: Path,
    model: AutoModel,
) -> None:
    wav, _ = librosa.load(str(audio_path), sr=cfg["data"]["audio"]["sr"])
    wav = wav[: wav.shape[0] - (wav.shape[0] % (cfg["data"]["audio"]["sr"] // 100))]
    wav = np.pad(
        wav,
        (0, int(cfg["data"]["audio"]["sr"] // 100 * 2)),
        mode="constant",
        constant_values=0,
    )
    wav_input = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        feature_extractor_output = model.feature_extractor(wav_input.cuda())
        feature_extractor_output = feature_extractor_output.permute(
            0, 2, 1
        )  # (B, T, C)
        feature_prj_output = model.feature_projection(
            feature_extractor_output
        )  # (B, T, C)
        encoder_output = model.encoder(
            feature_prj_output
        ).last_hidden_state  # (B, T, C)
    feature_prj_output_ndarray = feature_prj_output.cpu().squeeze(0).numpy()  # (T, C)
    encoder_output_ndarray = encoder_output.cpu().squeeze(0).numpy()  # (T, C)
    hubert_feature_prj_output_path.parent.mkdir(parents=True, exist_ok=True)
    hubert_encoder_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(hubert_feature_prj_output_path), feature_prj_output_ndarray)
    np.save(str(hubert_encoder_output_path), encoder_output_ndarray)


def save_numerical_features_kablab(
    cfg: omegaconf.DictConfig, model: AutoModel, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    audio_dir = Path(cfg["path"]["kablab"]["audio_dir"]).expanduser()
    hubert_feature_prj_output_dir = Path(
        cfg["path"]["kablab"]["hubert_feature_prj_output_dir"]
    ).expanduser()
    hubert_encoder_output_dir = Path(
        cfg["path"]["kablab"]["hubert_encoder_output_dir"]
    ).expanduser()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = audio_dir / row["speaker"] / f'{row["filename"]}.wav'
        hubert_feature_prj_output_path = (
            hubert_feature_prj_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        hubert_encoder_output_path = (
            hubert_encoder_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        if not audio_path.exists():
            continue
        process_save_numerical_features(
            cfg=cfg,
            audio_path=audio_path,
            hubert_feature_prj_output_path=hubert_feature_prj_output_path,
            hubert_encoder_output_path=hubert_encoder_output_path,
            model=model,
        )
        if debug:
            break


def save_numerical_features_hifi_captain(
    cfg: omegaconf.DictConfig,
    model: AutoModel,
    debug: bool,
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["hifi_captain"]["df_path"]).expanduser()))
    audio_dir = Path(cfg["path"]["hifi_captain"]["audio_dir"]).expanduser()
    hubert_feature_prj_output_dir = Path(
        cfg["path"]["hifi_captain"]["hubert_feature_prj_output_dir"]
    ).expanduser()
    hubert_encoder_output_dir = Path(
        cfg["path"]["hifi_captain"]["hubert_encoder_output_dir"]
    ).expanduser()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = (
            audio_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.wav'
        )
        hubert_feature_prj_output_path = (
            hubert_feature_prj_output_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.npy'
        )
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.npy'
        )
        if not audio_path.exists():
            continue
        process_save_numerical_features(
            cfg=cfg,
            audio_path=audio_path,
            hubert_feature_prj_output_path=hubert_feature_prj_output_path,
            hubert_encoder_output_path=hubert_encoder_output_path,
            model=model,
        )
        if debug:
            break


def save_numerical_features_jvs(
    cfg: omegaconf.DictConfig, model: AutoModel, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["jvs"]["df_path"]).expanduser()))
    audio_dir = Path(cfg["path"]["jvs"]["audio_dir"]).expanduser()
    hubert_feature_prj_output_dir = Path(
        cfg["path"]["jvs"]["hubert_feature_prj_output_dir"]
    ).expanduser()
    hubert_encoder_output_dir = Path(
        cfg["path"]["jvs"]["hubert_encoder_output_dir"]
    ).expanduser()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = (
            audio_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.wav'
        )
        hubert_feature_prj_output_path = (
            hubert_feature_prj_output_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.npy'
        )
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.npy'
        )
        if not audio_path.exists():
            continue
        process_save_numerical_features(
            cfg=cfg,
            audio_path=audio_path,
            hubert_feature_prj_output_path=hubert_feature_prj_output_path,
            hubert_encoder_output_path=hubert_encoder_output_path,
            model=model,
        )
        if debug:
            break


def save_numerical_features_jsut(
    cfg: omegaconf.DictConfig, model: AutoModel, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["jsut"]["df_path"]).expanduser()))
    audio_dir = Path(cfg["path"]["jsut"]["audio_dir"]).expanduser()
    hubert_feature_prj_output_dir = Path(
        cfg["path"]["jsut"]["hubert_feature_prj_output_dir"]
    ).expanduser()
    hubert_encoder_output_dir = Path(
        cfg["path"]["jsut"]["hubert_encoder_output_dir"]
    ).expanduser()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = audio_dir / row["dirname"] / "wav" / f'{row["filename"]}.wav'
        hubert_feature_prj_output_path = (
            hubert_feature_prj_output_dir
            / row["dirname"]
            / "wav"
            / f'{row["filename"]}.npy'
        )
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["dirname"]
            / "wav"
            / f'{row["filename"]}.npy'
        )
        if not audio_path.exists():
            continue
        process_save_numerical_features(
            cfg=cfg,
            audio_path=audio_path,
            hubert_feature_prj_output_path=hubert_feature_prj_output_path,
            hubert_encoder_output_path=hubert_encoder_output_path,
            model=model,
        )
        if debug:
            break


def save_numerical_features(cfg: omegaconf.DictConfig) -> None:
    """
    hubert特徴量の保存（連続値）
    """
    model = AutoModel.from_pretrained(cfg["model"]["decoder"]["hubert"]["model_name"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    save_numerical_features_kablab(cfg, model, debug=False)
    save_numerical_features_hifi_captain(cfg, model, debug=False)
    save_numerical_features_jvs(cfg, model, debug=False)
    save_numerical_features_jsut(cfg, model, debug=False)


def process_save_kmeans(
    cfg: omegaconf.DictConfig, hubert_encoder_output_list: list
) -> MiniBatchKMeans:
    hubert_encoder_output_all = np.concatenate(hubert_encoder_output_list, axis=0)
    kmeans = MiniBatchKMeans(
        n_clusters=cfg["model"]["decoder"]["hubert"]["n_clusters"],
        init="k-means++",
        batch_size=10000,
        compute_labels=False,
        random_state=42,
        verbose=1,
        max_iter=10000,
    ).fit(hubert_encoder_output_all)
    return kmeans


def save_kmeans_kablab(cfg: omegaconf.DictConfig) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    df = df.loc[(df["data_split"] == "train") & (df["corpus"] == "ATR")].reset_index(
        drop=True
    )
    hubert_encoder_output_dir = Path(
        cfg["path"]["kablab"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_encoder_output_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        hubert_encoder_output = np.load(str(hubert_encoder_output_path))
        hubert_encoder_output_list.append(hubert_encoder_output)

    kmeans = process_save_kmeans(
        cfg=cfg,
        hubert_encoder_output_list=hubert_encoder_output_list,
    )

    kmeans_dir = Path(cfg["path"]["kablab"]["hubert_kmeans_dir"]).expanduser()
    kmeans_path = (
        kmeans_dir / f"{cfg['model']['decoder']['hubert']['n_clusters']}.pickle"
    )
    kmeans_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(kmeans_path), "wb") as f:
        pickle.dump(kmeans, f)


def save_kmeans_hifi_captain(cfg: omegaconf.DictConfig) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["hifi_captain"]["df_path"]).expanduser()))
    df = (
        df.loc[df["data_split"] == "train"]
        .groupby(["speaker"])
        .apply(lambda x: x.sample(frac=0.5, random_state=42), include_groups=False)
        .reset_index()
        .drop(columns=["level_1"])
    )
    hubert_encoder_output_dir = Path(
        cfg["path"]["hifi_captain"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_encoder_output_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        hubert_encoder_output = np.load(str(hubert_encoder_output_path))
        hubert_encoder_output_list.append(hubert_encoder_output)

    kmeans = process_save_kmeans(
        cfg=cfg,
        hubert_encoder_output_list=hubert_encoder_output_list,
    )

    kmeans_dir = Path(cfg["path"]["hifi_captain"]["hubert_kmeans_dir"]).expanduser()
    kmeans_path = (
        kmeans_dir / f"{cfg['model']['decoder']['hubert']['n_clusters']}.pickle"
    )
    kmeans_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(kmeans_path), "wb") as f:
        pickle.dump(kmeans, f)


def save_kmeans_jsut(cfg: omegaconf.DictConfig) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["jsut"]["df_path"]).expanduser()))
    hubert_encoder_output_dir = Path(
        cfg["path"]["jsut"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_encoder_output_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["dirname"]
            / "wav"
            / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        hubert_encoder_output = np.load(str(hubert_encoder_output_path))
        hubert_encoder_output_list.append(hubert_encoder_output)

    kmeans = process_save_kmeans(
        cfg=cfg,
        hubert_encoder_output_list=hubert_encoder_output_list,
    )

    kmeans_dir = Path(cfg["path"]["jsut"]["hubert_kmeans_dir"]).expanduser()
    kmeans_path = (
        kmeans_dir / f"{cfg['model']['decoder']['hubert']['n_clusters']}.pickle"
    )
    kmeans_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(kmeans_path), "wb") as f:
        pickle.dump(kmeans, f)


def save_kmeans(cfg: omegaconf.DictConfig):
    save_kmeans_kablab(cfg)
    # save_kmeans_hifi_captain(cfg)
    # save_kmeans_jsut(cfg)


def process_save_clusters(
    hubert_encoder_output_path: Path,
    hubert_cluster_path: Path,
    kmeans: MiniBatchKMeans,
) -> None:
    hubert_encoder_output = np.load(str(hubert_encoder_output_path))
    hubert_cluster = kmeans.predict(hubert_encoder_output)
    hubert_cluster += 1  # 0はパディングに使いたいので、1加算しておく
    hubert_cluster_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(hubert_cluster_path), hubert_cluster)


def save_clusters_kablab(
    cfg: omegaconf.DictConfig, kmeans: MiniBatchKMeans, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    hubert_encoder_output_dir = Path(
        cfg["path"]["kablab"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_cluster_dir = Path(cfg["path"]["kablab"]["hubert_cluster_dir"]).expanduser()
    hubert_cluster_dir = (
        hubert_cluster_dir
        / cfg["model"]["decoder"]["hubert"]["kmeans"]
        / str(cfg["model"]["decoder"]["hubert"]["n_clusters"])
    )
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        hubert_cluster_path = (
            hubert_cluster_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        process_save_clusters(
            hubert_encoder_output_path=hubert_encoder_output_path,
            hubert_cluster_path=hubert_cluster_path,
            kmeans=kmeans,
        )
        if debug:
            break


def save_clusters_hifi_captain(
    cfg: omegaconf.DictConfig, kmeans: MiniBatchKMeans, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["hifi_captain"]["df_path"]).expanduser()))
    hubert_encoder_output_dir = Path(
        cfg["path"]["hifi_captain"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_cluster_dir = Path(
        cfg["path"]["hifi_captain"]["hubert_cluster_dir"]
    ).expanduser()
    hubert_cluster_dir = (
        hubert_cluster_dir
        / cfg["model"]["decoder"]["hubert"]["kmeans"]
        / str(cfg["model"]["decoder"]["hubert"]["n_clusters"])
    )
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.npy'
        )
        hubert_cluster_path = (
            hubert_cluster_dir
            / row["speaker"]
            / "wav"
            / row["parent_dir"]
            / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        process_save_clusters(
            hubert_encoder_output_path=hubert_encoder_output_path,
            hubert_cluster_path=hubert_cluster_path,
            kmeans=kmeans,
        )
        if debug:
            break


def save_clusters_jvs(
    cfg: omegaconf.DictConfig, kmeans: MiniBatchKMeans, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["jvs"]["df_path"]).expanduser()))
    hubert_encoder_output_dir = Path(
        cfg["path"]["jvs"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_cluster_dir = Path(cfg["path"]["jvs"]["hubert_cluster_dir"]).expanduser()
    hubert_cluster_dir = (
        hubert_cluster_dir
        / cfg["model"]["decoder"]["hubert"]["kmeans"]
        / str(cfg["model"]["decoder"]["hubert"]["n_clusters"])
    )
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.npy'
        )
        hubert_cluster_path = (
            hubert_cluster_dir
            / row["speaker"]
            / row["data"]
            / "wav24kHz16bit"
            / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        process_save_clusters(
            hubert_encoder_output_path=hubert_encoder_output_path,
            hubert_cluster_path=hubert_cluster_path,
            kmeans=kmeans,
        )
        if debug:
            break


def save_clusters_jsut(
    cfg: omegaconf.DictConfig, kmeans: MiniBatchKMeans, debug: bool
) -> None:
    df = pd.read_csv(str(Path(cfg["path"]["jsut"]["df_path"]).expanduser()))
    hubert_encoder_output_dir = Path(
        cfg["path"]["jsut"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_cluster_dir = Path(cfg["path"]["jsut"]["hubert_cluster_dir"]).expanduser()
    hubert_cluster_dir = (
        hubert_cluster_dir
        / cfg["model"]["decoder"]["hubert"]["kmeans"]
        / str(cfg["model"]["decoder"]["hubert"]["n_clusters"])
    )
    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_encoder_output_path = (
            hubert_encoder_output_dir
            / row["dirname"]
            / "wav"
            / f'{row["filename"]}.npy'
        )
        hubert_cluster_path = (
            hubert_cluster_dir / row["dirname"] / "wav" / f'{row["filename"]}.npy'
        )
        if not hubert_encoder_output_path.exists():
            continue
        process_save_clusters(
            hubert_encoder_output_path=hubert_encoder_output_path,
            hubert_cluster_path=hubert_cluster_path,
            kmeans=kmeans,
        )
        if debug:
            break


def save_clusters(cfg: omegaconf.DictConfig):
    """
    学習済みkmeansモデルを読み込み、クラスタリングした結果を保存する
    """
    kmeans_dir = Path(
        cfg["path"][cfg["model"]["decoder"]["hubert"]["kmeans"]]["hubert_kmeans_dir"]
    ).expanduser()
    kmeans_path = (
        kmeans_dir / f"{cfg['model']['decoder']['hubert']['n_clusters']}.pickle"
    )
    with open(str(kmeans_path), "rb") as f:
        kmeans = pickle.load(f)

    save_clusters_kablab(cfg, kmeans, debug=False)
    save_clusters_hifi_captain(cfg, kmeans, debug=False)
    save_clusters_jvs(cfg, kmeans, debug=False)
    # save_clusters_jsut(cfg, kmeans, debug=False)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    # save_numerical_features(cfg)
    save_kmeans(cfg)
    # save_clusters(cfg)


if __name__ == "__main__":
    main()
