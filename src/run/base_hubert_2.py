import subprocess
import time
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_with_retry(command: list[str], checkpoint_dir: Path, max_retries: int = 10):
    count = 0
    last_checkpoint_path_when_start = get_last_checkpoint_path(checkpoint_dir)
    resume = False
    while count < max_retries:
        try:
            if resume:
                subprocess.run(
                    command
                    + [
                        f"training.wandb.group_name=retry_{count}",
                        "training.resume=true",
                        f"training.resume_checkpoint_path={get_last_checkpoint_path(checkpoint_dir)}",
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    command + [f"training.wandb.group_name=retry_{count}"],
                    check=True,
                )
            break
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}. Retrying... ({count + 1}/{max_retries})")
            count += 1
            if last_checkpoint_path_when_start != get_last_checkpoint_path(
                checkpoint_dir
            ):
                resume = True
            time.sleep(5)


def run_hifigan(
    hifigan_checkpoint_dir: Path,
    hifigan_input: str,
    layer_index_cluster: int,
    n_clusters: int,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_base_hubert_2.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            "training=hifigan_base_hubert_2_debug"
            if debug
            else "training=hifigan_base_hubert_2",
            "training.finetune=false",
        ],
        hifigan_checkpoint_dir,
    )
    hifigan_checkpoint_path_hificaptain = str(
        get_last_checkpoint_path(hifigan_checkpoint_dir)
    )
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_base_hubert_2.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=true",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            "training=hifigan_base_hubert_2_debug"
            if debug
            else "training=hifigan_base_hubert_2",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(hifigan_checkpoint_path_hificaptain)}",
        ],
        hifigan_checkpoint_dir,
    )
    hifigan_checkpoint_path_jvs = str(get_last_checkpoint_path(hifigan_checkpoint_dir))
    return hifigan_checkpoint_path_jvs


def run_avhubert(
    hifigan_model_path_mel: str,
    hifigan_model_path_mel_speech_ssl: str,
    freeze_pattern: list[str],
    cluster_loss_weight: float,
    layer_index_cluster: int,
    n_clusters: int,
    lip2sp_checkpoint_dir: Path,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/base_hubert_2.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            "model.hifigan.freeze=true",
            f"model.hifigan.model_path_mel={hifigan_model_path_mel}",
            f"model.hifigan.model_path_mel_speech_ssl={hifigan_model_path_mel_speech_ssl}",
            f"model.freeze={freeze_pattern}",
            "model.decoder.vocoder_input=avhubert",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            "training=base_hubert_2_debug" if debug else "training=base_hubert_2",
            "training.loss_weights.mel_loss=1.0",
            "training.loss_weights.ssl_conv_feature_loss=1.0",
            f"training.loss_weights.ssl_feature_cluster_loss={cluster_loss_weight}",
            "training.loss_weights.mel_speech_ssl_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_speech_ssl_loss=0.0",
            "training.loss_weights.mel_ensemble_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=0.0",
            "training.finetune=false",
        ],
        lip2sp_checkpoint_dir,
    )
    checkpoint_path = str(get_last_checkpoint_path(lip2sp_checkpoint_dir))
    return checkpoint_path


def run_hubert(
    hifigan_model_path_mel: str,
    hifigan_model_path_mel_speech_ssl: str,
    freeze_pattern: list[str],
    cluster_loss_weight: float,
    layer_index_cluster: int,
    n_clusters: int,
    partial_update_use: bool,
    partial_update_lower_or_upper: str,
    partial_udpate_thres: int,
    lip2sp_checkpoint_dir: Path,
    finetune_start_model_path: Path,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/base_hubert_2.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            "model.hifigan.freeze=true",
            f"model.hifigan.model_path_mel={hifigan_model_path_mel}",
            f"model.hifigan.model_path_mel_speech_ssl={hifigan_model_path_mel_speech_ssl}",
            f"model.freeze={freeze_pattern}",
            "model.decoder.vocoder_input=speech_ssl",
            f"model.decoder.speech_ssl.partial_update.use={partial_update_use}",
            f"model.decoder.speech_ssl.partial_update.lower_or_upper={partial_update_lower_or_upper}",
            f"model.decoder.speech_ssl.partial_update.thres={partial_udpate_thres}",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            "training=base_hubert_2_debug" if debug else "training=base_hubert_2",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_loss=0.0",
            "training.loss_weights.mel_speech_ssl_loss=1.0",
            f"training.loss_weights.ssl_feature_cluster_speech_ssl_loss={cluster_loss_weight}",
            "training.loss_weights.mel_ensemble_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=0.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(finetune_start_model_path)}",
            "training.optimizer.learning_rate=5.0e-4",
        ],
        lip2sp_checkpoint_dir,
    )
    checkpoint_path = str(get_last_checkpoint_path(lip2sp_checkpoint_dir))
    return checkpoint_path


def run_ensemble(
    hifigan_model_path_mel: str,
    hifigan_model_path_mel_speech_ssl: str,
    freeze_pattern: list[str],
    cluster_loss_weight: float,
    layer_index_cluster: int,
    n_clusters: int,
    lip2sp_checkpoint_dir: Path,
    finetune_start_model_path: Path,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/base_hubert_2.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            "model.hifigan.freeze=true",
            f"model.hifigan.model_path_mel={hifigan_model_path_mel}",
            f"model.hifigan.model_path_mel_speech_ssl={hifigan_model_path_mel_speech_ssl}",
            f"model.freeze={freeze_pattern}",
            "model.decoder.vocoder_input=ensemble",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            "training=base_hubert_2_debug" if debug else "training=base_hubert_2",
            f"training.finetune_start_model_path={str(finetune_start_model_path)}",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_loss=0.0",
            "training.loss_weights.mel_speech_ssl_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_speech_ssl_loss=0.0",
            "training.loss_weights.mel_ensemble_loss=1.0",
            f"training.loss_weights.ssl_feature_cluster_ensemble_loss={cluster_loss_weight}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(finetune_start_model_path)}",
        ],
        lip2sp_checkpoint_dir,
    )
    checkpoint_path = str(get_last_checkpoint_path(lip2sp_checkpoint_dir))
    return checkpoint_path


def main():
    debug = False
    if debug:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_hifigan_base_hubert_2"
        ).expanduser()
        lip2sp_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_base_hubert_2"
        ).expanduser()
    else:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/hifigan_base_hubert_2"
        ).expanduser()
        lip2sp_checkpoint_dir = Path("~/lip2sp/checkpoints/base_hubert_2").expanduser()

    freeze_patterns = {
        "train_avhubert": [
            "ssl_model_encoder",
            "decoders_hubert",
            "ensemble_encoder",
            "decoders_ensemble",
        ],
        "train_hubert": [
            "avhubert",
            "decoders_avhubert",
            "ensemble_encoder",
            "decoders_ensemble",
        ],
        "train_ensemble": [
            "avhubert",
            "decoders_avhubert",
            "ssl_model_encoder",
            "decoders_hubert",
        ],
    }
    layer_index_cluster_lst = [8, 10, 12]
    n_clusters_lst = [100, 200]
    cluster_loss_weights = [0, 0.0001, 0.001, 0.01]

    for layer_index_cluster in layer_index_cluster_lst:
        for n_cluster in n_clusters_lst:
            if layer_index_cluster == 8 and n_cluster == 100:
                hifigan_checkpoint_path_jvs_mel = "/home/minami/lip2sp/checkpoints/hifigan_base_hubert_2/20240924_041029/epoch:23-step:31152.ckpt"
                hifigan_checkpoint_path_jvs_mel_speech_ssl = "/home/minami/lip2sp/checkpoints/hifigan_base_hubert_2/20240924_222851/epoch:17-step:23364.ckpt"
            else:
                hifigan_checkpoint_path_jvs_mel = run_hifigan(
                    hifigan_checkpoint_dir=hifigan_checkpoint_dir,
                    hifigan_input=["mel"],
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    debug=debug,
                )
                hifigan_checkpoint_path_jvs_mel_speech_ssl = run_hifigan(
                    hifigan_checkpoint_dir=hifigan_checkpoint_dir,
                    hifigan_input=["mel", "hubert_layer_feature_cluster"],
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    debug=debug,
                )

            for cluster_loss_weight in cluster_loss_weights:
                avhubert_checkpoint_path = run_avhubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_avhubert"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    debug=debug,
                )
                hubert_checkpoint_path = run_hubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_hubert"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    partial_update_use=False,
                    partial_update_lower_or_upper="",
                    partial_udpate_thres=0,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=avhubert_checkpoint_path,
                    debug=debug,
                )
                hubert_checkpoint_path = run_ensemble(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_ensemble"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=hubert_checkpoint_path,
                    debug=debug,
                )


if __name__ == "__main__":
    main()
