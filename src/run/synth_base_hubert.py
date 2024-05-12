import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_conv_decoder(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_path: Path,
    hifigan_input: str,
    group_name: str,
    conv_output_hubert_encoder_loss: float,
    conv_output_hubert_cluster_loss: float,
    learning_rate: float,
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
            f"model.hifigan.model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir)) if not hifigan_model_path.exists() else hifigan_model_path}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.hubert.freeze=true",
            "model.decoder.hubert.encoder_input_mask.use=false",
            "model.decoder.vocoder_input_cluster=conv",
            f"training.optimizer.learning_rate={learning_rate}",
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
            "training.loss_weights.hubert_output_mel_loss=0.0",
            "training.finetune=false",
            f"training.finetune_start_model_path={base_hubert_checkpoint_path}",
        ]
    )


def run_hubert(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    base_hubert_checkpoint_dir: Path,
    base_hubert_checkpoint_path: Path,
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
            "training.finetune=false",
            f"training.finetune_start_model_path={base_hubert_checkpoint_path}",
        ]
    )


def main():
    debug = True
    hifigan_model_path = {
        "feature": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240425_070203/epoch:26-step:35100.ckpt"
        ),
        "feature_hubert_encoder": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240509_021443/epoch:22-step:29900.ckpt"
        ),
        "feature_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240511_154553/epoch:17-step:23400.ckpt"
        ),
        "cat_mel_hubert_encoder": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240426_035112/epoch:19-step:26000.ckpt"
        ),
        "cat_mel_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240427_004201/epoch:28-step:37700.ckpt"
        ),
        "cat_hubert_encoder_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240510_132627/epoch:26-step:35100.ckpt"
        ),
        "cat_mel_hubert_encoder_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240429_040204/epoch:19-step:26000.ckpt"
        ),
    }
    hifigan_script_path = Path("/home/minami/lip2sp/src/main/hifigan.py")
    hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/hifigan")
    base_hubert_script_path = Path("/home/minami/lip2sp/src/synthesis/base_hubert.py")
    base_hubert_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/base_hubert")

    # 読み込めないやつ
    # feature
    # cat_mel_hubert_encoder

    run_cfg_list_conv = [
        # {
        #     "hifigan_input": "feature_hubert_encoder",
        #     "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240503_232759/epoch:46-step:2350.ckpt",
        # },
        # {
        #     "hifigan_input": "feature_hubert_cluster",
        #     "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240507_205750/epoch:44-step:2250.ckpt",
        # },
        # {
        #     "hifigan_input": "cat_hubert_encoder_hubert_cluster",
        #     "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240508_024744/epoch:44-step:2250.ckpt",
        # },
    ]
    for run_cfg in run_cfg_list_conv:
        run_conv_decoder(
            hifigan_script_path=hifigan_script_path,
            hifigan_checkpoint_dir=hifigan_checkpoint_dir,
            hifigan_model_path=hifigan_model_path[run_cfg["hifigan_input"]],
            base_hubert_script_path=base_hubert_script_path,
            base_hubert_checkpoint_path=run_cfg["base_hubert_checkpoint_path"],
            hifigan_input=run_cfg["hifigan_input"],
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=0.0,
            conv_output_hubert_cluster_loss=0.0,
            learning_rate=1.0e-3,
            debug=debug,
        )

    run_cfg_list_hubert = [
        # {
        #     "hifigan_input": "feature_hubert_encoder",
        #     "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240508_065428/epoch:4-step:250.ckpt",
        # },
        {
            "hifigan_input": "feature_hubert_cluster",
            "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240508_073151/epoch:6-step:350.ckpt",
        },
        {
            "hifigan_input": "cat_hubert_encoder_hubert_cluster",
            "base_hubert_checkpoint_path": "/home/minami/lip2sp/checkpoints/base_hubert/20240511_204947/epoch:21-step:1100.ckpt",
        },
    ]
    for run_cfg in run_cfg_list_hubert:
        run_hubert(
            hifigan_model_path=hifigan_model_path[run_cfg["hifigan_input"]],
            base_hubert_script_path=base_hubert_script_path,
            base_hubert_checkpoint_dir=base_hubert_checkpoint_dir,
            base_hubert_checkpoint_path=run_cfg["base_hubert_checkpoint_path"],
            hifigan_input=run_cfg["hifigan_input"],
            group_name="proposed",
            learning_rate=5.0e-4,
            hubert_output_reg_masked_loss=0.0,
            hubert_output_reg_unmasked_loss=0.0,
            hubert_output_reg_loss=0.0,
            hubert_output_cls_masked_loss=1.0,
            hubert_output_cls_unmasked_loss=0.0,
            hubert_output_cls_loss=1.0,
            encoder_input_mask_use=False,
            debug=debug,
        )


if __name__ == "__main__":
    main()
