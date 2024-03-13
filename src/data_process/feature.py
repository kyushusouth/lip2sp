import librosa
import numpy as np
import omegaconf
from python_speech_features import logfbank


def log10(x: np.array, eps: float) -> np.array:
    """
    epsでクリッピングした上で、常用対数をとる
    """
    return np.log10(np.maximum(x, eps))


def wav2mel(
    wav: np.array, cfg: omegaconf.DictConfig, ref_max: bool = False
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


def wav2mel_avhubert(wav: np.array, cfg: omegaconf.DictConfig) -> np.array:
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
