from pathlib import Path

from src.run.base_hubert_2 import run_avhubert, run_hubert


def main():
    debug = False
    if debug:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_hifigan_base_hubert_2"
        ).expanduser()
        lip2sp_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/debug_base_hubert_2"
        ).expanduser()
    else:
        hifigan_checkpoint_dir = Path(
            "~/lip2sp/checkpoints/hifigan_base_hubert_2"
        ).expanduser()
        lip2sp_checkpoint_dir = Path("~/lip2sp/checkpoints/base_hubert_2").expanduser()

    freeze_patterns = {
        "train_avhubert": [
            "ssl_model_encoder",
            "decoders_hubert",
            "ensemble_encoder",
            "decoders_ensemble",
        ],
        "train_hubert": [
            "avhubert",
            "decoders_avhubert",
            "ensemble_encoder",
            "decoders_ensemble",
        ],
        "train_ensemble": [
            "avhubert",
            "decoders_avhubert",
            "ssl_model_encoder",
            "decoders_hubert",
        ],
        "e2e": [
            "ensemble_encoder",
            "decoders_ensemble",
        ],
    }

    # run_hubertを使って、training.is_only_synthesis=trueかつtraining.is_val_synthesis=true
    hifigan_checkpoint_path_jvs_mel = ""
    hifigan_checkpoint_path_jvs_mel_speech_ssl = "/home/minami/lip2sp/checkpoints/hifigan_base_hubert_2/20241023_060247/epoch:27-step:36344.ckpt"
    # baseline
    run_avhubert(
        hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
        hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
        freeze_pattern=freeze_patterns["train_avhubert"],
        mel_loss_weight=1.0,
        ssl_conv_feature_loss_weight=0.0,
        cluster_loss_weight=0.001,
        layer_index_cluster=8,
        n_clusters=100,
        lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
        is_finetuning=True,
        finetune_start_model_path=Path(
            "/home/minami/lip2sp/checkpoints/base_hubert_2/20241028_013251/epoch:47-step:2400.ckpt"
        ),
        is_strict=True,
        is_only_synthesis=True,
        is_val_synthesis=True,
        debug=debug,
    )
    # networkB(Not-Pretrained)
    run_hubert(
        hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
        hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
        freeze_pattern=freeze_patterns["train_hubert"],
        cluster_loss_weight=0.1,
        layer_index_cluster=8,
        n_clusters=100,
        partial_update_use=False,
        partial_update_lower_or_upper="",
        partial_udpate_thres=0,
        speech_ssl_load_pretrained_weight=False,
        lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
        finetune_start_model_path=Path(
            "/home/minami/lip2sp/checkpoints/base_hubert_2/20241028_200713/epoch:24-step:1250.ckpt"
        ),
        is_finetuning=True,
        speech_ssl_input_type="ssl_conv_feature",
        is_strict=True,
        learning_rate=5.0e-4,
        is_only_synthesis=True,
        is_val_synthesis=True,
        debug=debug,
    )
    # networkB(Not-Pretrained, A-SingleTask)
    run_hubert(
        hifigan_model_path_mel=hifigan_checkpoint_path_jvs_mel,
        hifigan_model_path_mel_speech_ssl=hifigan_checkpoint_path_jvs_mel_speech_ssl,
        freeze_pattern=freeze_patterns["train_hubert"],
        cluster_loss_weight=0.1,
        layer_index_cluster=8,
        n_clusters=100,
        partial_update_use=False,
        partial_update_lower_or_upper="",
        partial_udpate_thres=0,
        speech_ssl_load_pretrained_weight=False,
        lip2sp_checkpoint_dir=lip2sp_checkpoint_dir,
        finetune_start_model_path=Path(
            "/home/minami/lip2sp/checkpoints/base_hubert_2/20241109_201211/epoch:17-step:900.ckpt"
        ),
        is_finetuning=True,
        speech_ssl_input_type="ssl_conv_feature",
        is_strict=True,
        learning_rate=5.0e-4,
        is_only_synthesis=True,
        is_val_synthesis=True,
        debug=debug,
    )


if __name__ == "__main__":
    main()
