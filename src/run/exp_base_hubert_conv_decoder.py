import argparse
import subprocess
from pathlib import Path


def run_conv_decoder(
    hifigan_model_path: Path,
    base_hubert_script_path: Path,
    hifigan_input: str,
    group_name: str,
    conv_output_hubert_encoder_loss: float,
    conv_output_hubert_cluster_loss: float,
    learning_rate: float,
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
            f"training.seed={seed}",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    debug = True
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
    loss_weight_list = [0.0001, 0.001, 0.01, 0.1, 1.0]

    run_conv_decoder(
        hifigan_model_path=hifigan_model_path["feature"],
        base_hubert_script_path=base_hubert_script_path,
        hifigan_input="feature",
        group_name="conv_decoder",
        conv_output_hubert_encoder_loss=0.0,
        conv_output_hubert_cluster_loss=0.0,
        learning_rate=1.0e-3,
        seed=args.seed,
        debug=debug,
    )
    run_conv_decoder(
        hifigan_model_path=hifigan_model_path["cat_mel_hubert_encoder"],
        base_hubert_script_path=base_hubert_script_path,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="conv_decoder",
        conv_output_hubert_encoder_loss=1.0,
        conv_output_hubert_cluster_loss=0.0,
        learning_rate=1.0e-3,
        seed=args.seed,
        debug=debug,
    )
    for loss_weight in loss_weight_list:
        run_conv_decoder(
            hifigan_model_path=hifigan_model_path["cat_mel_hubert_cluster"],
            base_hubert_script_path=base_hubert_script_path,
            hifigan_input="cat_mel_hubert_cluster",
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=0.0,
            conv_output_hubert_cluster_loss=loss_weight,
            learning_rate=1.0e-3,
            seed=args.seed,
            debug=debug,
        )
    for loss_weight in loss_weight_list:
        run_conv_decoder(
            hifigan_model_path=hifigan_model_path[
                "cat_mel_hubert_encoder_hubert_cluster"
            ],
            base_hubert_script_path=base_hubert_script_path,
            hifigan_input="cat_mel_hubert_encoder_hubert_cluster",
            group_name="conv_decoder",
            conv_output_hubert_encoder_loss=1.0,
            conv_output_hubert_cluster_loss=loss_weight,
            learning_rate=1.0e-3,
            seed=args.seed,
            debug=debug,
        )


if __name__ == "__main__":
    main()
