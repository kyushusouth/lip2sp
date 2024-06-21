import argparse
import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_avhubert(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    hifigan_input: str,
    group_name: str,
    learning_rate: float,
    seed: int,
    debug: bool,
) -> Path:
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
            f"training.optimizer.learning_rate={learning_rate}",
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
            "training.loss_weights.hubert_output_mel_loss=0.0",
            "training.finetune=false",
            f"training.seed={seed}",
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
    learning_rate: float,
    hubert_output_reg_masked_loss: float,
    hubert_output_reg_unmasked_loss: float,
    hubert_output_reg_loss: float,
    hubert_output_cls_masked_loss: float,
    hubert_output_cls_unmasked_loss: float,
    hubert_output_cls_loss: float,
    encoder_input_mask_use: bool,
    seed: int,
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
            f"model.decoder.hubert.encoder_input_mask.use={encoder_input_mask_use}",
            "model.decoder.vocoder_input_cluster=hubert",
            f"training.optimizer.learning_rate={learning_rate}",
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
            "training.loss_weights.hubert_output_mel_loss=0.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={trained_avhubert_path}",
            f"training.seed={seed}",
        ]
    )
    return get_last_checkpoint_path(base_hubert_checkpoint_dir)


def run_hubert_all_finetuning(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_path: Path,
    trained_avhubert_path: Path,
    hifigan_input: str,
    group_name: str,
    learning_rate: float,
    hubert_output_reg_masked_loss: float,
    hubert_output_reg_unmasked_loss: float,
    hubert_output_reg_loss: float,
    hubert_output_cls_masked_loss: float,
    hubert_output_cls_unmasked_loss: float,
    hubert_output_cls_loss: float,
    encoder_input_mask_use: bool,
    seed: int,
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
            "model.decoder.hubert.freeze=false",
            f"model.decoder.hubert.encoder_input_mask.use={encoder_input_mask_use}",
            "model.decoder.vocoder_input_cluster=hubert",
            f"training.optimizer.learning_rate={learning_rate}",
            "training=base_hubert_debug" if debug else "training=base_hubert",
            f"training.wandb.group_name={group_name}",
            "training.loss_weights.conv_output_mel_loss=1.0",
            "training.loss_weights.conv_output_hubert_encoder_loss=0.0",
            "training.loss_weights.conv_output_hubert_cluster_loss=0.0",
            "training.loss_weights.conv_output_hubert_prj_loss=0.0",
            f"training.loss_weights.hubert_output_reg_masked_loss={hubert_output_reg_masked_loss}",
            f"training.loss_weights.hubert_output_reg_unmasked_loss={hubert_output_reg_unmasked_loss}",
            f"training.loss_weights.hubert_output_reg_loss={hubert_output_reg_loss}",
            f"training.loss_weights.hubert_output_cls_masked_loss={hubert_output_cls_masked_loss}",
            f"training.loss_weights.hubert_output_cls_unmasked_loss={hubert_output_cls_unmasked_loss}",
            f"training.loss_weights.hubert_output_cls_loss={hubert_output_cls_loss}",
            "training.loss_weights.hubert_output_mel_loss=0.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={base_hubert_checkpoint_path}",
            f"training.seed={seed}",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    debug = False
    # hifigan_model_path = {
    #     "feature": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240425_070203/epoch:26-step:35100.ckpt"
    #     ),
    #     "feature_hubert_encoder": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240509_021443/epoch:22-step:29900.ckpt"
    #     ),
    #     "feature_hubert_cluster": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240511_154553/epoch:17-step:23400.ckpt"
    #     ),
    #     "cat_mel_hubert_encoder": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240426_035112/epoch:19-step:26000.ckpt"
    #     ),
    #     "cat_mel_hubert_cluster": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240427_004201/epoch:28-step:37700.ckpt"
    #     ),
    #     "cat_hubert_encoder_hubert_cluster": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240510_132627/epoch:26-step:35100.ckpt"
    #     ),
    #     "cat_mel_hubert_encoder_hubert_cluster": Path(
    #         "/home/minami/lip2sp/checkpoints/hifigan/20240429_040204/epoch:19-step:26000.ckpt"
    #     ),
    # }
    hifigan_model_path = {
        "feature": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240618_133442/epoch:19-step:26000.ckpt"
        ),
        "feature_hubert_encoder": Path(""),
        "feature_hubert_cluster": Path(""),
        "cat_mel_hubert_encoder": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240619_092235/epoch:26-step:35100.ckpt"
        ),
        "cat_mel_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240620_050910/epoch:23-step:31200.ckpt"
        ),
        "cat_hubert_encoder_hubert_cluster": Path(""),
        "cat_mel_hubert_encoder_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240621_005533/epoch:19-step:26000.ckpt"
        ),
    }
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")
    if debug:
        base_hubert_checkpoint_dir = Path(
            "/home/minami/lip2sp/checkpoints/debug_base_hubert"
        )
    loss_weight_list = [0.0001, 0.001, 0.01, 0.1, 1.0]

    # trained_avhubert_path = Path(
    #     "/home/minami/lip2sp/checkpoints/base_hubert/20240520_032120/epoch:35-step:1800.ckpt"
    # )

    trained_avhubert_path = run_avhubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        learning_rate=1.0e-3,
        seed=args.seed,
        debug=debug,
    )
    base_hubert_last_checkpoint_path = run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        learning_rate=5.0e-4,
        hubert_output_reg_masked_loss=1.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=1.0,
        hubert_output_cls_masked_loss=0.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=0.0,
        encoder_input_mask_use=False,
        seed=args.seed,
        debug=debug,
    )
    run_hubert_all_finetuning(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_path=base_hubert_last_checkpoint_path,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="proposed",
        learning_rate=5.0e-4,
        hubert_output_reg_masked_loss=1.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=1.0,
        hubert_output_cls_masked_loss=0.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=0.0,
        encoder_input_mask_use=False,
        seed=args.seed,
        debug=debug,
    )
    base_hubert_last_checkpoint_path = run_hubert(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="proposed",
        learning_rate=5.0e-4,
        hubert_output_reg_masked_loss=0.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=0.0,
        hubert_output_cls_masked_loss=1.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=1.0,
        encoder_input_mask_use=False,
        seed=args.seed,
        debug=debug,
    )
    run_hubert_all_finetuning(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
        base_hubert_script_path=base_hubert_script_path,
        base_hubert_checkpoint_path=base_hubert_last_checkpoint_path,
        trained_avhubert_path=trained_avhubert_path,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="proposed",
        learning_rate=5.0e-4,
        hubert_output_reg_masked_loss=0.0,
        hubert_output_reg_unmasked_loss=0.0,
        hubert_output_reg_loss=0.0,
        hubert_output_cls_masked_loss=1.0,
        hubert_output_cls_unmasked_loss=0.0,
        hubert_output_cls_loss=1.0,
        encoder_input_mask_use=False,
        seed=args.seed,
        debug=debug,
    )
    for loss_weight in loss_weight_list:
        base_hubert_last_checkpoint_path = run_hubert(
            hifigan_model_path=hifigan_model_path[
                "cat_mel_hubert_encoder_hubert_cluster"
            ],
            base_hubert_script_path=base_hubert_script_path,
            base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
            trained_avhubert_path=trained_avhubert_path,
            hifigan_input="cat_mel_hubert_encoder_hubert_cluster",
            group_name="proposed",
            learning_rate=5.0e-4,
            hubert_output_reg_masked_loss=1.0,
            hubert_output_reg_unmasked_loss=0.0,
            hubert_output_reg_loss=1.0,
            hubert_output_cls_masked_loss=1.0,
            hubert_output_cls_unmasked_loss=0.0,
            hubert_output_cls_loss=loss_weight,
            encoder_input_mask_use=False,
            seed=args.seed,
            debug=debug,
        )
        run_hubert_all_finetuning(
            hifigan_model_path=hifigan_model_path[
                "cat_mel_hubert_encoder_hubert_cluster"
            ],
            base_hubert_script_path=base_hubert_script_path,
            base_hubert_checkpoint_path=base_hubert_last_checkpoint_path,
            trained_avhubert_path=trained_avhubert_path,
            hifigan_input="cat_mel_hubert_encoder_hubert_cluster",
            group_name="proposed",
            learning_rate=5.0e-4,
            hubert_output_reg_masked_loss=1.0,
            hubert_output_reg_unmasked_loss=0.0,
            hubert_output_reg_loss=1.0,
            hubert_output_cls_masked_loss=1.0,
            hubert_output_cls_unmasked_loss=0.0,
            hubert_output_cls_loss=loss_weight,
            encoder_input_mask_use=False,
            seed=args.seed,
            debug=debug,
        )


if __name__ == "__main__":
    main()
