import subprocess
from pathlib import Path




def baseline(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    debug: bool,
    mel_aug: bool,
) -> None:
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            "model.hifigan.input=feature",
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.wandb.group_name=baseline",
            "training.finetune=false",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            "model.hifigan.input=feature",
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.wandb.group_name=baseline",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            "model.hifigan.input=feature",
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            "model.decoder.hubert.freeze=true",
            "model.decoder.vocoder_input_cluster=conv",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=baseline",
            "training.finetune=false",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            "training.loss_weights.hubert_output_reg_loss=0.0",
            "training.loss_weights.hubert_output_cls_loss=0.0",
        ]
    )


def speech_units(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    debug: bool,
    mel_aug: bool,
) -> None:
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            "model.hifigan.input=cat_mel_hubert_cluster",
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.wandb.group_name=speech_units",
            "training.finetune=false",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            "model.hifigan.input=cat_mel_hubert_cluster",
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.wandb.group_name=speech_units",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            "model.hifigan.input=cat_mel_hubert_cluster",
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            "model.decoder.hubert.freeze=true",
            "model.decoder.vocoder_input_cluster=conv",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=speech_units",
            "training.finetune=false",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.1",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            "training.loss_weights.hubert_output_reg_loss=0.0",
            "training.loss_weights.hubert_output_cls_loss=0.0",
        ]
    )


def proposed(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    debug: bool,
    mel_aug: bool,
    hifigan_input: str,
) -> None:
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=true",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.finetune=false",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            "training=hifigan_debug" if debug else "training=hifigan",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            f"training.augs.add_gaussian_noise.use={mel_aug}",
            f"training.augs.spec_gaussian_blur.use={mel_aug}",
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
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            "model.decoder.hubert.freeze=true",
            "model.decoder.vocoder_input_cluster=hubert",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.finetune=false",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_loss=1.0",
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
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir))}",
            "model.decoder.hubert.freeze=false",
            "model.decoder.vocoder_input_cluster=hubert",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            "training.wandb.group_name=proposed",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(base_hubert_checkpoint_dir))}",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0",
            "training.loss_weights.conv_output_hubert_prj_loss=1.0",
            "training.loss_weights.hubert_output_reg_loss=1.0",
            "training.loss_weights.hubert_output_cls_loss=1.0",
            "training.optimizer.learning_rate=1.0e-4",
        ]
    )


def main():
    debug = False
    hifigan_script_path = Path("/home/minami/lip2sp/src/main/hifigan.py")
    hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/hifigan")
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")

    # baseline(
    #     hifigan_script_path=hifigan_script_path,
    #     hifigan_checkpoint_dir=hifigan_checkpoint_dir,
    #     base_hubert_script_path=base_hubert_script_path,
    #     base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
    #     debug=debug,
    #     mel_aug=True,
    # )
    # speech_units(
    #     hifigan_script_path=hifigan_script_path,
    #     hifigan_checkpoint_dir=hifigan_checkpoint_dir,
    #     base_hubert_script_path=base_hubert_script_path,
    #     base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
    #     debug=debug,
    #     mel_aug=True,
    # )
    proposed(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        debug=debug,
        mel_aug=False,
        hifigan_input="cat_mel_hubert_encoder",
    )
    proposed(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        debug=debug,
        mel_aug=True,
        hifigan_input="cat_mel_hubert_encoder",
    )
    proposed(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        debug=debug,
        mel_aug=False,
        hifigan_input="cat_mel_hubert_cluster",
    )
    proposed(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        debug=debug,
        mel_aug=True,
        hifigan_input="cat_mel_hubert_cluster",
    )


if __name__ == "__main__":
    main()
