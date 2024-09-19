import subprocess
import time
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_with_retry(command: list[str], max_retries: int = 10):
    count = 0
    while count < max_retries:
        try:
            subprocess.run(
                command + [f"training.wandb.group_name=retry_{count}"],
                check=True,
            )
            break
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}. Retrying... ({count + 1}/{max_retries})")
            count += 1
            time.sleep(5)


def run_hifigan(
    hifigan_checkpoint_dir: Path,
    hifigan_input: str,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_speech_memory.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_speech_memory_debug"
            if debug
            else "training=hifigan_speech_memory",
            "training.finetune=false",
        ]
    )
    hifigan_checkpoint_path_hificaptain = str(
        get_last_checkpoint_path(hifigan_checkpoint_dir)
    )
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_speech_memory.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=true",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_speech_memory_debug"
            if debug
            else "training=hifigan_speech_memory",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(hifigan_checkpoint_path_hificaptain)}",
        ]
    )
    hifigan_checkpoint_path_jvs = str(get_last_checkpoint_path(hifigan_checkpoint_dir))
    return hifigan_checkpoint_path_jvs


def run_avhubert(
    hifigan_input: str,
    hifigan_checkpoint_path_jvs: str,
    loss_weights_ssl_feature_cluster_linear_loss: float,
    n_clusters: int,
    layer_index_cluster: int,
    memory_atten_use: bool,
    debug: bool,
) -> None:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/speech_memory.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            f"model.hifigan.model_path={hifigan_checkpoint_path_jvs}",
            f"model.decoder.speech_ssl.n_clusters={n_clusters}",
            f"model.decoder.speech_ssl.layer_index_cluster={layer_index_cluster}",
            f"model.memory_atten.use={memory_atten_use}",
            "training=speech_memory_debug" if debug else "training=speech_memory",
            "training.loss_weights.mel_loss=1.0",
            f"training.loss_weights.ssl_feature_cluster_linear_loss={loss_weights_ssl_feature_cluster_linear_loss}",
            "training.finetune=false",
        ]
    )


def main():
    debug = False
    if debug:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_hifigan_speech_memory"
        ).expanduser()
    else:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/hifigan_speech_memory"
        ).expanduser()

    n_clusters_lst = [100, 200]
    layer_index_cluster_lst = [8, 10, 12]
    loss_weight_cluster_lst = [0.001, 0.01, 0.1]

    hifigan_checkpoint_path_jvs = run_hifigan(
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        hifigan_input=["mel"],
        debug=debug,
    )

    for n_clusters in n_clusters_lst:
        for layer_index_cluster in layer_index_cluster_lst:
            for loss_weight_cluster in loss_weight_cluster_lst:
                run_avhubert(
                    hifigan_input=["mel"],
                    hifigan_checkpoint_path_jvs=str(hifigan_checkpoint_path_jvs),
                    loss_weights_ssl_feature_cluster_linear_loss=loss_weight_cluster,
                    n_clusters=n_clusters,
                    layer_index_cluster=layer_index_cluster,
                    memory_atten_use=False,
                    debug=debug,
                )
                run_avhubert(
                    hifigan_input=["mel"],
                    hifigan_checkpoint_path_jvs=str(hifigan_checkpoint_path_jvs),
                    loss_weights_ssl_feature_cluster_linear_loss=loss_weight_cluster,
                    n_clusters=n_clusters,
                    layer_index_cluster=layer_index_cluster,
                    memory_atten_use=True,
                    debug=debug,
                )


if __name__ == "__main__":
    main()
