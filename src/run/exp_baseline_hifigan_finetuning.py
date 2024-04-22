import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    base_hubert_hifigan_finetuning_script_path: Path,
    hifigan_input: str,
    group_name: str,
    debug: bool,
):
    # subprocess.run(
    #     [
    #         "python",
    #         str(hifigan_script_path),
    #         "data_choice.kablab.use=false",
    #         "data_choice.jvs.use=false",
    #         "data_choice.hifi_captain.use=true",
    #         "data_choice.jsut.use=false",
    #         f"model.hifigan.input={hifigan_input}",
    #         "model.hifigan.freeze=false",
    #         "training=hifigan_debug" if debug else "training=hifigan",
    #         f"training.wandb.group_name={group_name}",
    #         "training.finetune=false",
    #     ]
    # )
    # subprocess.run(
    #     [
    #         "python",
    #         str(hifigan_script_path),
    #         "data_choice.kablab.use=false",
    #         "data_choice.jvs.use=true",
    #         "data_choice.hifi_captain.use=false",
    #         "data_choice.jsut.use=false",
    #         f"model.hifigan.input={hifigan_input}",
    #         "model.hifigan.freeze=false",
    #         "training=hifigan_debug" if debug else "training=hifigan",
    #         f"training.wandb.group_name={group_name}",
    #         "training.finetune=true",
    #         f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
    #     ]
    # )
    # subprocess.run(
    #     [
    #         "python",
    #         str(base_hubert_script_path),
    #         "data_choice.kablab.use=true",
    #         "data_choice.jvs.use=false",
    #         "data_choice.hifi_captain.use=false",
    #         "data_choice.jsut.use=false",
    #         f"model.hifigan.input={hifigan_input}",
    #         f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
    #         "model.hifigan.freeze=true",
    #         "model.avhubert.freeze=false",
    #         "model.spk_emb_layer.freeze=false",
    #         "model.decoder.conv.freeze=false",
    #         "model.decoder.hubert.freeze=true",
    #         "model.decoder.hubert.encoder_input_mask.use=false",
    #         "model.decoder.vocoder_input_cluster=conv",
    #         "training.optimizer.learning_rate=1.0e-3",
    #         "training=base_hubert_debug" if debug else "training=base_hubert",
    #         f"training.wandb.group_name={group_name}",
    #         "training.loss_weights.conv_output_mel_loss=1.0",
    #         "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
    #         "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
    #         "training.loss_weights.conv_output_hubert_prj_loss=0.0",
    #         "training.loss_weights.hubert_output_reg_masked_loss=0.0",
    #         "training.loss_weights.hubert_output_reg_unmasked_loss=0.0",
    #         "training.loss_weights.hubert_output_reg_loss=0.0",
    #         "training.loss_weights.hubert_output_cls_masked_loss=0.0",
    #         "training.loss_weights.hubert_output_cls_unmasked_loss=0.0",
    #         "training.loss_weights.hubert_output_cls_loss=0.0",
    #         "training.finetune=false",
    #     ]
    # )
    subprocess.run(
        [
            "python",
            str(base_hubert_hifigan_finetuning_script_path),
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.hubert.freeze=true",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=conv",
            "training=base_hubert_hifigan_finetuning_debug"
            if debug
            else "training=base_hubert_hifigan_finetuning",
            f"training.wandb.group_name={group_name}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(base_hubert_checkpoint_dir))}",
        ]
    )


def main():
    debug = True
    hifigan_script_path = Path("/home/minami/lip2sp/src/main/hifigan.py")
    hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/hifigan")
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    base_hubert_hifigan_finetuning_script_path = Path(
        "/home/minami/lip2sp/src/main/base_hubert_hifigan_finetuning.py"
    )
    if debug:
        hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/debug_hifigan")
        base_hubert_checkpoint_dir = Path(
            "/home/minami/lip2sp/checkpoints/debug_base_hubert"
        )
    run(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        base_hubert_hifigan_finetuning_script_path=base_hubert_hifigan_finetuning_script_path,
        hifigan_input="feature",
        group_name="baseline",
        debug=debug,
    )


if __name__ == "__main__":
    main()
