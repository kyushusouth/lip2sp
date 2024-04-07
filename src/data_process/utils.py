import librosa
import numpy as np
import omegaconf
from python_speech_features import logfbank


def log10(x: np.ndarray, eps: float) -> np.ndarray:
    """
    epsでクリッピングした上で、常用対数をとる
    """
    return np.log10(np.maximum(x, eps))


def get_upsample(cfg: omegaconf.DictConfig) -> int:
    """
    動画のfpsと音響特徴量のフレームあたりの秒数から対応関係を求める
    """
    n_mel_frames_per_sec = (
        cfg["data"]["audio"]["sr"] // cfg["data"]["audio"]["hop_length"]
    )
    upsample = n_mel_frames_per_sec // cfg["data"]["video"]["fps"]
    return upsample


def get_upsample_hubert(cfg: omegaconf.DictConfig) -> int:
    """
    動画のfpsとHuBERT特徴量の対応関係を返す
    """
    upsample = 50 // cfg["data"]["video"]["fps"]
    return upsample


def wav2mel(
    wav: np.ndarray, cfg: omegaconf.DictConfig, ref_max: bool = False
) -> np.ndarray:
    """
    音声波形をメルスペクトログラムに変換
    wav : (T,)
    feature : (C, T)
    """
    feature = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg["data"]["audio"]["sr"],
        n_fft=cfg["data"]["audio"]["n_fft"],
        hop_length=cfg["data"]["audio"]["hop_length"],
        win_length=cfg["data"]["audio"]["win_length"],
        window="hann",
        n_mels=cfg["data"]["audio"]["n_mels"],
        fmin=cfg["data"]["audio"]["f_min"],
        fmax=cfg["data"]["audio"]["f_max"],
    )
    if ref_max:
        feature = librosa.power_to_db(feature, ref=np.max)
    else:
        feature = log10(feature, eps=cfg["data"]["audio"]["eps"])
    return feature


def wav2mel_avhubert(wav: np.ndarray, cfg: omegaconf.DictConfig) -> np.ndarray:
    """
    音声波形をavhubertで用いられる音響特徴量に変換
    wav : (T,)
    feature : (C, T)
    """
    feature = logfbank(
        wav,
        samplerate=cfg["data"]["audio"]["sr"],
        winlen=cfg["data"]["audio"]["win_length"] / cfg["data"]["audio"]["sr"],
        winstep=cfg["data"]["audio"]["hop_length"] / cfg["data"]["audio"]["sr"],
        nfft=cfg["data"]["audio"]["n_fft"],
        nfilt=cfg["data"]["audio"]["avhubert_nfilt"],
        preemph=cfg["data"]["audio"]["avhubert_preemph"],
        lowfreq=cfg["data"]["audio"]["f_min"],
        highfreq=cfg["data"]["audio"]["f_max"],
    ).T  # (C, T)
    feature = feature.astype(np.float32)
    return feature
