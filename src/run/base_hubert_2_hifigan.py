from pathlib import Path

from src.run.base_hubert_2 import run_hifigan


def main():
    debug = True
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
    }
    layer_index_cluster_lst = [8]
    n_clusters_lst = [100]
    cluster_loss_weights = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for layer_index_cluster in layer_index_cluster_lst:
        for n_cluster in n_clusters_lst:
            hifigan_checkpoint_path_jvs_mel_speech_ssl = run_hifigan(
                hifigan_checkpoint_dir=hifigan_checkpoint_dir,
                hifigan_input=["mel", "hubert_layer_feature_cluster"],
                layer_index_cluster=layer_index_cluster,
                n_clusters=n_cluster,
                debug=debug,
            )


if __name__ == "__main__":
    main()
