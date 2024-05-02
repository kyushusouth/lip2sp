import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_hifigan(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    hifigan_input: str,
    group_name: str,
    debug: bool,
):
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_debug" if debug else "training=hifigan",
            f"training.wandb.group_name={group_name}",
            "training.finetune=false",
        ]
    )
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=true",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_debug" if debug else "training=hifigan",
            f"training.wandb.group_name={group_name}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
        ]
    )


def run_conv_decoder(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    hifigan_input: str,
    group_name: str,
    conv_output_hubert_encoder_loss: float,
    conv_output_hubert_cluster_loss: float,
    debug: bool,
):
    if not hifigan_model_path.exists():
        run_hifigan(
            hifigan_script_path=hifigan_script_path,
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_input=hifigan_input,
            group_name=group_name,
            debug=debug,
        )
    subprocess.run(
        [
            "python",
            str(base_hubert_script_path),
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir)) if not hifigan_model_path.exists() else hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=true",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=conv",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=1.0",
            f"training.loss_weights.conv_output_hubert_encoder_loss={conv_output_hubert_encoder_loss}",
            f"training.loss_weights.conv_output_hubert_cluster_loss={conv_output_hubert_cluster_loss}",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            "training.loss_weights.hubert_output_reg_masked_loss=0.0",
            "training.loss_weights.hubert_output_reg_unmasked_loss=0.0",
            "training.loss_weights.hubert_output_reg_loss=0.0",
            "training.loss_weights.hubert_output_cls_masked_loss=0.0",
            "training.loss_weights.hubert_output_cls_unmasked_loss=0.0",
            "training.loss_weights.hubert_output_cls_loss=0.0",
            "training.finetune=false",
        ]
    )


def main():
    debug = False
    hifigan_model_path = {
        "cat_mel_hubert_encoder": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240426_035112/epoch:19-step:26000.ckpt"
        ),
        "cat_mel_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240427_004201/epoch:28-step:37700.ckpt"
        ),
        "cat_mel_hubert_encoder_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240429_040204/epoch:19-step:26000.ckpt"
        ),
    }
    hifigan_script_path = Path("/home/minami/lip2sp/src/main/hifigan.py")
    hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/hifigan")
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    if debug:
        hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/debug_hifigan")
    loss_weight_list = [0.1, 0.5, 1.0]

    for loss_weight in loss_weight_list:
        run_conv_decoder(
            hifigan_script_path=hifigan_script_path,
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
            base_hubert_script_path=base_hubert_script_path,
            hifigan_input="cat_mel_hubert_encoder",
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=loss_weight,
            conv_output_hubert_cluster_loss=0.0,
            debug=debug,
        )
    for loss_weight in loss_weight_list:
        run_conv_decoder(
            hifigan_script_path=hifigan_script_path,
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
            base_hubert_script_path=base_hubert_script_path,
            hifigan_input="cat_mel_hubert_cluster",
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=0.0,
            conv_output_hubert_cluster_loss=loss_weight,
            debug=debug,
        )
    for loss_weight in loss_weight_list:
        run_conv_decoder(
            hifigan_script_path=hifigan_script_path,
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_model_path=hifigan_model_path[
                "cat_mel_hubert_encoder_hubert_cluster"
            ],
            base_hubert_script_path=base_hubert_script_path,
            hifigan_input="cat_mel_hubert_encoder_hubert_cluster",
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=loss_weight,
            conv_output_hubert_cluster_loss=loss_weight,
            debug=debug,
        )


if __name__ == "__main__":
    main()
