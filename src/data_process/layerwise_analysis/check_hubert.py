import hydra
import pickle
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import omegaconf
import textgrids
import torch
from tqdm import tqdm
from transformers import AutoModel


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    audio_path = "/home/minami/dataset/lip/wav/F02_kablab/ATR503_i44.wav"
    wav, _ = librosa.load(str(audio_path), sr=cfg.data.audio.sr)
    wav, _ = librosa.load(str(audio_path), sr=cfg.data.audio.sr)
    wav = wav[: wav.shape[0] - (wav.shape[0] % (cfg.data.audio.sr // 100))]
    wav = np.pad(
        wav,
        (0, int(cfg.data.audio.sr // 100 * 2)),
        mode="constant",
        constant_values=0,
    )
    wav_input = torch.from_numpy(wav).unsqueeze(0).cuda()

    hubert = AutoModel.from_pretrained(cfg.model.decoder.hubert.model_name).cuda()
    feature_extractor_output = hubert.feature_extractor(wav_input.cuda())
    feature_extractor_output = feature_extractor_output.permute(0, 2, 1)  # (B, T, C)
    feature_prj_output = hubert.feature_projection(
        feature_extractor_output
    )  # (B, T, C)

    encoder_output = hubert.encoder.pos_conv_embed(feature_prj_output)
    encoder_output = hubert.encoder.layer_norm(encoder_output)
    encoder_output = hubert.encoder.dropout(encoder_output)
    transformer_layer_outputs = []
    transformer_layer_output = encoder_output
    for layer in hubert.encoder.layers:
        transformer_layer_output = layer(transformer_layer_output)
        transformer_layer_output = transformer_layer_output[0]
        transformer_layer_outputs.append(transformer_layer_output)

    output_2 = hubert.encoder(feature_prj_output).last_hidden_state

    output = hubert(
        wav_input, output_attentions=False, output_hidden_states=True, return_dict=True
    )

    print(torch.equal(output.last_hidden_state, output.hidden_states[-1]))


if __name__ == "__main__":
    main()
