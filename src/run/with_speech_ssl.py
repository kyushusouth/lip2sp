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
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_multiple_ssl_debug"
            if debug
            else "training=hifigan_multiple_ssl",
            "training.finetune=false",
        ]
    )
    hifigan_checkpoint_path_hificaptain = str(
        get_last_checkpoint_path(hifigan_checkpoint_dir)
    )
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl.py",
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=true",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_multiple_ssl_debug"
            if debug
            else "training=hifigan_multiple_ssl",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(hifigan_checkpoint_path_hificaptain)}",
        ]
    )
    hifigan_checkpoint_path_jvs = str(get_last_checkpoint_path(hifigan_checkpoint_dir))
    return hifigan_checkpoint_path_jvs


def run_avhubert(
    hifigan_input: str,
    lip2sp_checkpoint_dir: Path,
    hifigan_checkpoint_path_jvs: str,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.linear.freeze=false",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=avhubert",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=1.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=1.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=0.0",
            "training.finetune=false",
        ]
    )
    avhubert_checkpoint_path = str(get_last_checkpoint_path(lip2sp_checkpoint_dir))
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=avhubert",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={avhubert_checkpoint_path}",
        ]
    )
    return avhubert_checkpoint_path


def run_speech_ssl(
    hifigan_input: str,
    speech_ssl_model_name: str,
    speech_ssl_n_clusters: int,
    speech_ssl_partial_update_use: bool,
    speech_ssl_partial_update_lower_limit_index: int,
    avhubert_checkpoint_path: str,
    lip2sp_checkpoint_dir: Path,
    hifigan_checkpoint_path_jvs: str,
    debug: bool,
) -> str:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=false",
            "model.decoder.ensemble.freeze=true",
            f"model.decoder.speech_ssl.model_name={speech_ssl_model_name}",
            f"model.decoder.speech_ssl.n_clusters={speech_ssl_n_clusters}",
            f"model.decoder.speech_ssl.partial_update.use={speech_ssl_partial_update_use}",
            f"model.decoder.speech_ssl.partial_update.lower_limit_index={speech_ssl_partial_update_lower_limit_index}",
            "model.decoder.vocoder_input_cluster=speech_ssl",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=1.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=0.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={avhubert_checkpoint_path}",
        ]
    )
    lip2sp_with_speech_ssl_checkpoint_path = str(
        get_last_checkpoint_path(lip2sp_checkpoint_dir)
    )
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=speech_ssl",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={lip2sp_with_speech_ssl_checkpoint_path}",
        ]
    )
    return lip2sp_with_speech_ssl_checkpoint_path


def run_ensemble(
    hifigan_input: str,
    speech_ssl_model_name: str,
    speech_ssl_n_clusters: int,
    lip2sp_with_speech_ssl_checkpoint_path: str,
    lip2sp_checkpoint_dir: Path,
    hifigan_checkpoint_path_jvs: str,
    debug: bool,
) -> None:
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=false",
            f"model.decoder.speech_ssl.model_name={speech_ssl_model_name}",
            f"model.decoder.speech_ssl.n_clusters={speech_ssl_n_clusters}",
            "model.decoder.speech_ssl.partial_update.use=false",
            "model.decoder.speech_ssl.partial_update.lower_limit_index=0",
            "model.decoder.vocoder_input_cluster=ensemble",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ensemble_loss=1.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={lip2sp_with_speech_ssl_checkpoint_path}",
        ]
    )
    lip2sp_with_speech_ssl_ensemble_checkpoint_path = str(
        get_last_checkpoint_path(lip2sp_checkpoint_dir)
    )
    run_with_retry(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=ensemble",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={lip2sp_with_speech_ssl_ensemble_checkpoint_path}",
        ]
    )


def main():
    debug = False
    if debug:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_hifigan_multiple_ssl"
        ).expanduser()
        lip2sp_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_with_speech_ssl"
        ).expanduser()
    else:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/hifigan_multiple_ssl"
        ).expanduser()
        lip2sp_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/with_speech_ssl"
        ).expanduser()

    hifigan_input_lst = [
        ["hubert_final_feature_cluster"],
        ["mel", "hubert_final_feature_cluster"],
    ]
    speech_ssl_model_name = "rinna/japanese-hubert-base"
    speech_ssl_n_clusters = 200

    for hifigan_input in hifigan_input_lst:
        hifigan_checkpoint_path_jvs = run_hifigan(
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_input=hifigan_input,
            debug=debug,
        )
        avhubert_checkpoint_path = run_avhubert(
            hifigan_input=hifigan_input,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )

        lip2sp_with_speech_ssl_checkpoint_path = run_speech_ssl(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            speech_ssl_partial_update_use=False,
            speech_ssl_partial_update_lower_limit_index=0,
            avhubert_checkpoint_path=avhubert_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )
        run_ensemble(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            lip2sp_with_speech_ssl_checkpoint_path=lip2sp_with_speech_ssl_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )

        lip2sp_with_speech_ssl_checkpoint_path = run_speech_ssl(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            speech_ssl_partial_update_use=True,
            speech_ssl_partial_update_lower_limit_index=9,
            avhubert_checkpoint_path=avhubert_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )
        run_ensemble(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            lip2sp_with_speech_ssl_checkpoint_path=lip2sp_with_speech_ssl_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )

        lip2sp_with_speech_ssl_checkpoint_path = run_speech_ssl(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            speech_ssl_partial_update_use=True,
            speech_ssl_partial_update_lower_limit_index=6,
            avhubert_checkpoint_path=avhubert_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )
        run_ensemble(
            hifigan_input=hifigan_input,
            speech_ssl_model_name=speech_ssl_model_name,
            speech_ssl_n_clusters=speech_ssl_n_clusters,
            lip2sp_with_speech_ssl_checkpoint_path=lip2sp_with_speech_ssl_checkpoint_path,
            lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
            hifigan_checkpoint_path_jvs=hifigan_checkpoint_path_jvs,
            debug=debug,
        )


if __name__ == "__main__":
    main()
