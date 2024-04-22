import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def proposed(
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    hifigan_input: str,
    hifigan_model_path: Path,
    hubert_output_reg_unmasked_loss: float,
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
            f"model.hifigan.model_path={str(hifigan_model_path)}",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=true",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.loss_weights.conv_output_mel_loss=1.0",
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
    subprocess.run(
        [
            "python",
            str(base_hubert_script_path),
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            f"model.hifigan.model_path={str(hifigan_model_path)}",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=true",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.loss_weights.conv_output_mel_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            "training.loss_weights.hubert_output_reg_masked_loss=1.0",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_masked_loss=1.0",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_reg_unmasked_loss}",
            "training.loss_weights.hubert_output_cls_loss=1.0",
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
            f"model.hifigan.model_path={str(hifigan_model_path)}",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-3",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.loss_weights.conv_output_mel_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            "training.loss_weights.hubert_output_reg_masked_loss=1.0",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_masked_loss=1.0",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_reg_unmasked_loss}",
            "training.loss_weights.hubert_output_cls_loss=1.0",
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
            f"model.hifigan.model_path={str(hifigan_model_path)}",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training.optimizer.learning_rate=1.0e-4",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            "training.loss_weights.hubert_output_reg_masked_loss=1.0",
            "training.loss_weights.hubert_output_reg_unmasked_loss=1.0",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_masked_loss=1.0",
            "training.loss_weights.hubert_output_cls_unmasked_loss=1.0",
            "training.loss_weights.hubert_output_cls_loss=1.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(base_hubert_checkpoint_dir))}",
        ]
    )


def compare_last_finetuning(
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    hifigan_input: str,
    hifigan_model_path: Path,
    hubert_output_reg_unmasked_loss: float,
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
            f"model.hifigan.model_path={str(hifigan_model_path)}",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=false",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            "training.loss_weights.hubert_output_reg_masked_loss=1.0",
            "training.loss_weights.hubert_output_reg_unmasked_loss=1.0",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_masked_loss=1.0",
            "training.loss_weights.hubert_output_cls_unmasked_loss=1.0",
            "training.loss_weights.hubert_output_cls_loss=1.0",
            "training.finetune=true",
            "training.finetune_start_model_path=/home/minami/lip2sp/checkpoints/base_hubert/20240422_125045/epoch:18-step:475.ckpt",
            "training.optimizer.learning_rate=1.0e-4",
        ]
    )


def main():
    debug = False
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    hifigan_cat_mel_hubert_encoder = Path(
        "/home/minami/lip2sp/checkpoints/hifigan/20240418_083428/epoch:27-step:36400.ckpt"
    )
    hifigan_cat_mel_hubert_cluster = Path(
        "/home/minami/lip2sp/checkpoints/hifigan/20240420_041048/epoch:12-step:16900.ckpt"
    )

    proposed(
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_encoder",
        hifigan_model_path=str(hifigan_cat_mel_hubert_encoder),
        hubert_output_reg_unmasked_loss=0.5,
        debug=debug,
    )
    proposed(
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_encoder",
        hifigan_model_path=str(hifigan_cat_mel_hubert_encoder),
        hubert_output_reg_unmasked_loss=1.0,
        debug=debug,
    )
    proposed(
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_cluster",
        hifigan_model_path=str(hifigan_cat_mel_hubert_cluster),
        hubert_output_reg_unmasked_loss=0.0,
        debug=debug,
    )
    proposed(
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_cluster",
        hifigan_model_path=str(hifigan_cat_mel_hubert_cluster),
        hubert_output_reg_unmasked_loss=0.5,
        debug=debug,
    )
    proposed(
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_cluster",
        hifigan_model_path=str(hifigan_cat_mel_hubert_cluster),
        hubert_output_reg_unmasked_loss=1.0,
        debug=debug,
    )


if __name__ == "__main__":
    main()
