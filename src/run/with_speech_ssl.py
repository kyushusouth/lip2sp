import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def main():
    hifigan_checkpoint_dir = Path(
        "~/lip2sp/checkpoints/debug_hifigan_multiple_ssl"
    ).expanduser()

    lip2sp_checkpoint_dir = Path(
        "~/lip2sp/checkpoints/debug_with_speech_ssl"
    ).expanduser()

    debug = True
    hifigan_input = "hubert_final_feature_cluster"
    speech_ssl_model_name = "rinna/japanese-hubert-base"
    speech_ssl_n_clusters = 200
    speech_ssl_partial_update_use = False
    speech_ssl_partial_update_lower_limit_index = 0

    # 原音声の音声ssl離散特徴量を入力とするhifiganの学習
    # subprocess.run(
    #     [
    #         "python",
    #         "/home/minami/lip2sp/src/main/hifigan_multiple_ssl.py",
    #         "data_choice.kablab.use=false",
    #         "data_choice.jvs.use=false",
    #         "data_choice.hifi_captain.use=true",
    #         "data_choice.jsut.use=false",
    #         f"model.hifigan.input={hifigan_input}",
    #         "model.hifigan.freeze=false",
    #         "training=hifigan_multiple_ssl_debug"
    #         if debug
    #         else "training=hifigan_multiple_ssl",
    #         "training.finetune=false",
    #     ]
    # )
    # hifigan_checkpoint_path_hificaptain = str(
    #     get_last_checkpoint_path(hifigan_checkpoint_dir)
    # )

    # subprocess.run(
    #     [
    #         "python",
    #         "/home/minami/lip2sp/src/main/hifigan_multiple_ssl.py",
    #         "data_choice.kablab.use=false",
    #         "data_choice.jvs.use=true",
    #         "data_choice.hifi_captain.use=false",
    #         "data_choice.jsut.use=false",
    #         f"model.hifigan.input={hifigan_input}",
    #         "model.hifigan.freeze=false",
    #         "training=hifigan_multiple_ssl_debug"
    #         if debug
    #         else "training=hifigan_multiple_ssl",
    #         "training.finetune=true",
    #         f"training.finetune_start_model_path={hifigan_checkpoint_path_hificaptain}",
    #     ]
    # )
    hifigan_checkpoint_path_jvs = str(get_last_checkpoint_path(hifigan_checkpoint_dir))

    # avhubertで音声ssl中間特徴量と音声ssl離散特徴量を推定する学習
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=false",
            "model.spk_emb_layer.freeze=false",
            "model.decoder.conv.freeze=false",
            "model.decoder.linear.freeze=false",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=avhubert",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=1.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=1.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=0.0",
            "training.finetune=false",
        ]
    )
    avhubert_checkpoint_path = str(get_last_checkpoint_path(lip2sp_checkpoint_dir))

    # 音声sslモデルを用いた音声ssl離散特徴量を推定する学習
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=false",
            "model.decoder.ensemble.freeze=true",
            f"model.decoder.speech_ssl.model_name={speech_ssl_model_name}",
            f"model.decoder.speech_ssl.n_clusters={speech_ssl_n_clusters}",
            f"model.decoder.speech_ssl.partial_update.use={speech_ssl_partial_update_use}",
            f"model.decoder.speech_ssl.partial_update.lower_limit_index={speech_ssl_partial_update_lower_limit_index}",
            "model.decoder.vocoder_input_cluster=speech_ssl",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=1.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={avhubert_checkpoint_path}",
        ]
    )
    lip2sp_with_speech_ssl_checkpoint_path = str(
        get_last_checkpoint_path(lip2sp_checkpoint_dir)
    )

    # avhubertとhubertのアンサンブル
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/with_speech_ssl.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=true",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=false",
            f"model.decoder.speech_ssl.model_name={speech_ssl_model_name}",
            f"model.decoder.speech_ssl.n_clusters={speech_ssl_n_clusters}",
            f"model.decoder.speech_ssl.partial_update.use={speech_ssl_partial_update_use}",
            f"model.decoder.speech_ssl.partial_update.lower_limit_index={speech_ssl_partial_update_lower_limit_index}",
            "model.decoder.vocoder_input_cluster=speech_ssl",
            "training=with_speech_ssl_debug" if debug else "training=with_speech_ssl",
            "training.loss_weights.mel_loss=0.0",
            "training.loss_weights.ssl_conv_feature_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_linear_loss=0.0",
            "training.loss_weights.ssl_feature_cluster_ssl_loss=1.0",
            "training.finetune=true",
            f"training.finetune_start_model_path={lip2sp_with_speech_ssl_checkpoint_path}",
        ]
    )
    lip2sp_with_speech_ssl_ensemble_checkpoint_path = str(
        get_last_checkpoint_path(lip2sp_checkpoint_dir)
    )

    # hifiganのfinetuningおよびテスト
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=avhubert",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={avhubert_checkpoint_path}",
        ]
    )
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=speech_ssl",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={lip2sp_with_speech_ssl_checkpoint_path}",
        ]
    )
    subprocess.run(
        [
            "python",
            "/home/minami/lip2sp/src/main/hifigan_multiple_ssl_finetuning.py",
            "data_choice.kablab.use=true",
            "data_choice.jvs.use=false",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "model.avhubert.freeze=true",
            "model.spk_emb_layer.freeze=true",
            "model.decoder.conv.freeze=true",
            "model.decoder.linear.freeze=true",
            "model.decoder.speech_ssl.freeze=true",
            "model.decoder.ensemble.freeze=true",
            "model.decoder.vocoder_input_cluster=ensemble",
            "training=hifigan_multiple_ssl_finetuning_debug"
            if debug
            else "training=hifigan_multiple_ssl_finetuning",
            "training.finetune=true",
            f"training.finetune_start_model_path={hifigan_checkpoint_path_jvs}",
            f"training.lip2sp_model_path={lip2sp_with_speech_ssl_ensemble_checkpoint_path}",
        ]
    )


if __name__ == "__main__":
    main()
