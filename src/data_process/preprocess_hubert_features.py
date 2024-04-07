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


def save_numerical_features(cfg: omegaconf.DictConfig):
    model = AutoModel.from_pretrained(cfg["model"]["decoder"]["hubert"]["model_name"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    df = df.loc[df["speaker"].isin(cfg["data_choice"]["kablab"]["speaker"])]
    df = df.loc[df["corpus"].isin(cfg["data_choice"]["kablab"]["corpus"])]
    audio_dir = Path(cfg["path"]["kablab"]["audio_dir"]).expanduser()
    audio_path_list = []
    for i, row in df.iterrows():
        audio_path = audio_dir / row["speaker"] / f'{row["filename"]}.wav'
        if not audio_path.exists():
            continue
        audio_path_list.append(audio_path)

    for audio_path in tqdm(audio_path_list):
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

        feature_prj_output_ndarray = (
            feature_prj_output.cpu().squeeze(0).numpy()
        )  # (T, C)
        encoder_output_ndarray = encoder_output.cpu().squeeze(0).numpy()  # (T, C)

        save_path_feature_prj_output = (
            Path(cfg["path"]["kablab"]["hubert_feature_prj_output_dir"]).expanduser()
            / audio_path.parents[0].name
            / f"{audio_path.stem}.npy"
        )
        save_path_feature_prj_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path_feature_prj_output), feature_prj_output_ndarray)

        save_path_encoder_output = (
            Path(cfg["path"]["kablab"]["hubert_encoder_output_dir"]).expanduser()
            / audio_path.parents[0].name
            / f"{audio_path.stem}.npy"
        )
        save_path_encoder_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path_encoder_output), encoder_output_ndarray)


def save_kmeans(cfg: omegaconf.DictConfig):
    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    df = df.loc[df["speaker"].isin(cfg["data_choice"]["kablab"]["speaker"])]
    df = df.loc[df["corpus"].isin(cfg["data_choice"]["kablab"]["corpus"])]
    df = df.loc[df["data_split"] == "train"]
    hubert_encoder_output_dir = Path(
        cfg["path"]["kablab"]["hubert_encoder_output_dir"]
    ).expanduser()
    hubert_feature_list = []
    for i, row in df.iterrows():
        hubert_path = (
            hubert_encoder_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        if not hubert_path.exists():
            continue
        hubert_feature = np.load(str(hubert_path))
        hubert_feature_list.append(hubert_feature)

    hubert_features = np.concatenate(hubert_feature_list, axis=0)
    kmeans = MiniBatchKMeans(
        n_clusters=cfg["model"]["decoder"]["hubert"]["n_clusters"],
        init="k-means++",
        batch_size=10000,
        compute_labels=False,
        random_state=42,
        verbose=1,
    ).fit(hubert_features)

    save_path = Path(cfg["path"]["kablab"]["hubert_kmeans_path"]).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(save_path), "wb") as f:
        pickle.dump(kmeans, f)


def save_cluster(cfg: omegaconf.DictConfig):
    df = pd.read_csv(str(Path(cfg["path"]["kablab"]["df_path"]).expanduser()))
    df = df.loc[df["speaker"].isin(cfg["data_choice"]["kablab"]["speaker"])]
    df = df.loc[df["corpus"].isin(cfg["data_choice"]["kablab"]["corpus"])]
    hubert_encoder_output_dir = Path(
        cfg["path"]["kablab"]["hubert_encoder_output_dir"]
    ).expanduser()

    kmeans_path = Path(cfg["path"]["kablab"]["hubert_kmeans_path"]).expanduser()
    with open(str(kmeans_path), "rb") as f:
        kmeans = pickle.load(f)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        hubert_path = (
            hubert_encoder_output_dir / row["speaker"] / f'{row["filename"]}.npy'
        )
        if not hubert_path.exists():
            continue
        hubert_feature = np.load(str(hubert_path))
        hubert_feature_cluster = kmeans.predict(hubert_feature)
        # 0はパディングに使いたいので、1加算しておく
        hubert_feature_cluster += 1
        save_path = (
            Path(cfg["path"]["kablab"]["hubert_cluster_dir"]).expanduser()
            / hubert_path.parents[0].name
            / f"{hubert_path.stem}.npy"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), hubert_feature_cluster)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    save_numerical_features(cfg)
    save_kmeans(cfg)
    save_cluster(cfg)


if __name__ == "__main__":
    main()
