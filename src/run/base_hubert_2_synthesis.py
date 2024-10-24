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
    ssl_conv_feature_loss_weight: float,
    cluster_loss_weight: float,
    layer_index_cluster: int,
    n_clusters: int,
    lip2sp_checkpoint_dir: Path,
    debug: bool,
    finetune_start_model_path: Path,
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
            f"training.loss_weights.ssl_conv_feature_loss={ssl_conv_feature_loss_weight}",
            f"training.loss_weights.ssl_feature_cluster_loss={cluster_loss_weight}",
            "training.loss_weights.mel_speech_ssl_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_speech_ssl_loss=0.0",
            "training.loss_weights.mel_ensemble_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=0.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={finetune_start_model_path}",
            "training.is_only_synthesis=true",
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
    speech_ssl_load_pretrained_weight: bool,
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
            f"model.decoder.speech_ssl.load_pretrained_weight={speech_ssl_load_pretrained_weight}",
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
            "training.is_only_synthesis=true",
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
            "training.is_only_synthesis=true",
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
    layer_index_cluster_lst = [8]
    n_clusters_lst = [100]
    cluster_loss_weights = [0.0001, 0.001, 0.01, 0.1, 1.0]

    checkpoints = {
        8: {
            100: {
                0.0001: {
                    0: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241006_222745/epoch:47-step:2400.ckpt",
                    1: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_100539/epoch:42-step:2150.ckpt",
                    2: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_120716/epoch:46-step:2350.ckpt",
                    3: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_133847/epoch:22-step:1150.ckpt",
                    4: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_143413/epoch:20-step:1050.ckpt",
                    5: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_153309/epoch:22-step:1150.ckpt",
                },
                0.001: {
                    0: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241007_003155/epoch:45-step:2300.ckpt",
                    1: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_162836/epoch:42-step:2150.ckpt",
                    2: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_183041/epoch:45-step:2300.ckpt",
                    3: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_200238/epoch:22-step:1150.ckpt",
                    4: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_205846/epoch:24-step:1250.ckpt",
                    5: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_220427/epoch:27-step:1400.ckpt",
                },
                0.01: {
                    0: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241007_023651/epoch:47-step:2400.ckpt",
                    1: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241004_233941/epoch:45-step:2300.ckpt",
                    2: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_011425/epoch:48-step:2450.ckpt",
                    3: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_024543/epoch:45-step:2300.ckpt",
                    4: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_040604/epoch:30-step:1550.ckpt",
                    5: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_052149/epoch:44-step:2250.ckpt",
                },
                0.1: {
                    0: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241007_044147/epoch:14-step:750.ckpt",
                    1: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_120810/epoch:26-step:1350.ckpt",
                    2: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_134056/epoch:32-step:1650.ckpt",
                    3: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_150112/epoch:18-step:950.ckpt",
                    4: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_155146/epoch:17-step:900.ckpt",
                    5: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_164631/epoch:6-step:350.ckpt",
                },
                1.0: {
                    0: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241007_054850/epoch:9-step:500.ckpt",
                    1: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_171958/epoch:9-step:500.ckpt",
                    2: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_181656/epoch:9-step:500.ckpt",
                    3: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_185910/epoch:14-step:750.ckpt",
                    4: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_194354/epoch:10-step:550.ckpt",
                    5: "/home/minami/lip2sp/checkpoints/base_hubert_2/20241005_202659/epoch:16-step:850.ckpt",
                },
            }
        }
    }

    for layer_index_cluster in layer_index_cluster_lst:
        for n_cluster in n_clusters_lst:
            hifigan_checkpoint_path_jvs_mel = None
            hifigan_checkpoint_path_jvs_mel_speech_ssl = "/home/minami/lip2sp/checkpoints/hifigan_base_hubert_2/20241023_060247/epoch:27-step:36344.ckpt"

            for cluster_loss_weight in cluster_loss_weights:
                run_avhubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_avhubert"],
                    ssl_conv_feature_loss_weight=0.0,
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    debug=debug,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][0],
                )

                run_avhubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_avhubert"],
                    ssl_conv_feature_loss_weight=1.0,
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    debug=debug,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][1],
                )

                run_hubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_hubert"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    partial_update_use=False,
                    partial_update_lower_or_upper="",
                    partial_udpate_thres=0,
                    speech_ssl_load_pretrained_weight=False,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][2],
                    debug=debug,
                )
                run_ensemble(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_ensemble"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][3],
                    debug=debug,
                )

                run_hubert(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_hubert"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    partial_update_use=False,
                    partial_update_lower_or_upper="",
                    partial_udpate_thres=0,
                    speech_ssl_load_pretrained_weight=True,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][4],
                    debug=debug,
                )
                run_ensemble(
                    hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
                    hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
                    freeze_pattern=freeze_patterns["train_ensemble"],
                    cluster_loss_weight=cluster_loss_weight,
                    layer_index_cluster=layer_index_cluster,
                    n_clusters=n_cluster,
                    lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
                    finetune_start_model_path=checkpoints[layer_index_cluster][
                        n_cluster
                    ][cluster_loss_weight][5],
                    debug=debug,
                )


if __name__ == "__main__":
    main()
