import omegaconf
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class LayerwiseASRModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained(cfg.model.layerwise_asr.model_name)

        self.lstm = nn.LSTM(
            cfg.model.layerwise_asr.hidden_channels,
            cfg.model.layerwise_asr.hidden_channels,
            num_layers=cfg.model.layerwise_asr.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(
            cfg.model.layerwise_asr.hidden_channels * 2,
            cfg.training.token_index.num_token,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_len: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """
        args:
            x : (B, T)
            feature_len : (B,)
            padding_mask: (B, T)
        return:
            out: (B, T, C)
        """
        encoder_output = self.encoder(
            input_values=x,
            attention_mask=padding_mask,
            mask_time_indices=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_state = encoder_output["hidden_states"][
            self.cfg.model.layerwise_asr.layer_index
        ]  # (B, T, C)

        feature_len = torch.clamp(feature_len, max=hidden_state.shape[1])

        hidden_state = pack_padded_sequence(
            hidden_state, feature_len.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (hn, cn) = self.lstm(hidden_state)
        out = pad_packed_sequence(out, batch_first=True)[0]
        out = self.fc(out)
        return out
