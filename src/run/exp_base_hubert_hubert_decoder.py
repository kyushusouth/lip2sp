import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_avhubert(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    hifigan_input: str,
    group_name: str,
    debug: bool,
):
    subprocess.run(
        [
            "python",
            str(base_hubert_script_path),
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            f"model.hifigan.model_path={hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=true",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            "training.loss_weights.hubert_output_reg_masked_loss=0.0",
            "training.loss_weights.hubert_output_reg_unmasked_loss=0.0",
            "training.loss_weights.hubert_output_reg_loss=0.0",
            "training.loss_weights.hubert_output_cls_masked_loss=0.0",
            "training.loss_weights.hubert_output_cls_unmasked_loss=0.0",
            "training.loss_weights.hubert_output_cls_loss=0.0",
            "training.finetune=false",
        ]
    )
    return get_last_checkpoint_path(base_hubert_checkpoint_dir)


def run_hubert(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    trained_avhubert_path: Path,
    hifigan_input: str,
    group_name: str,
    hubert_output_reg_masked_loss: float,
    hubert_output_reg_unmasked_loss: float,
    hubert_output_reg_loss: float,
    hubert_output_cls_masked_loss: float,
    hubert_output_cls_unmasked_loss: float,
    hubert_output_cls_loss: float,
    debug: bool,
):
    subprocess.run(
        [
            "python",
            str(base_hubert_script_path),
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            f"model.hifigan.model_path={hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=true",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=0.0",
            "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            f"training.loss_weights.hubert_output_reg_masked_loss={hubert_output_reg_masked_loss}",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            f"training.loss_weights.hubert_output_reg_loss={hubert_output_reg_loss}",
            f"training.loss_weights.hubert_output_cls_masked_loss={hubert_output_cls_masked_loss}",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_cls_unmasked_loss}",
            f"training.loss_weights.hubert_output_cls_loss={hubert_output_cls_loss}",
            "training.finetune=true",
            f"training.finetune_start_model_path={trained_avhubert_path}",
        ]
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
            f"model.hifigan.model_path={hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=0.0",
            "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            f"training.loss_weights.hubert_output_reg_masked_loss={hubert_output_reg_masked_loss}",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            f"training.loss_weights.hubert_output_reg_loss={hubert_output_reg_loss}",
            f"training.loss_weights.hubert_output_cls_masked_loss={hubert_output_cls_masked_loss}",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_cls_unmasked_loss}",
            f"training.loss_weights.hubert_output_cls_loss={hubert_output_cls_loss}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(base_hubert_checkpoint_dir))}",
        ]
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
            f"model.hifigan.model_path={hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            f"training.loss_weights.hubert_output_reg_masked_loss={hubert_output_reg_masked_loss}",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            f"training.loss_weights.hubert_output_reg_loss={hubert_output_reg_loss}",
            f"training.loss_weights.hubert_output_cls_masked_loss={hubert_output_cls_masked_loss}",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_cls_unmasked_loss}",
            f"training.loss_weights.hubert_output_cls_loss={hubert_output_cls_loss}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(base_hubert_checkpoint_dir))}",
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
    }
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    if debug:
        base_hubert_checkpoint_dir = Path(
            "/home/minami/lip2sp/checkpoints/debug_base_hubert"
        )

    trained_avhubert_path = run_avhubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        debug=debug,
    )

    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        hubert_output_reg_masked_loss=1.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=1.0,
        hubert_output_cls_masked_loss=0.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=0.0,
        debug=debug,
    )
    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        hubert_output_reg_masked_loss=1.0,
        hubert_output_reg_unmasked_loss=0.5,
        hubert_output_reg_loss=1.0,
        hubert_output_cls_masked_loss=0.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=0.0,
        debug=debug,
    )
    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        hubert_output_reg_masked_loss=1.0,
        hubert_output_reg_unmasked_loss=1.0,
        hubert_output_reg_loss=1.0,
        hubert_output_cls_masked_loss=0.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=0.0,
        debug=debug,
    )

    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="proposed",
        hubert_output_reg_masked_loss=0.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=0.0,
        hubert_output_cls_masked_loss=1.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=1.0,
        debug=debug,
    )
    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="proposed",
        hubert_output_reg_masked_loss=0.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=0.0,
        hubert_output_cls_masked_loss=1.0,
        hubert_output_cls_unmasked_loss=0.5,
        hubert_output_cls_loss=1.0,
        debug=debug,
    )
    run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="proposed",
        hubert_output_reg_masked_loss=0.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=0.0,
        hubert_output_cls_masked_loss=1.0,
        hubert_output_cls_unmasked_loss=1.0,
        hubert_output_cls_loss=1.0,
        debug=debug,
    )


if __name__ == "__main__":
    main()
