import omegaconf
import torch


def adjust_seq_lengths(batch: list, cfg: omegaconf.DictConfig) -> tuple:
    (
        wav,
        wav_len,
        feature_len,
        token,
        token_len,
        speaker,
        filename,
    ) = list(zip(*batch))

    max_len_wav = max([w.shape[0] for w in wav])
    wav_padded = []
    for w in wav:
        w_padded = torch.zeros(max_len_wav)
        w_padded[: w.shape[0]] = w
        wav_padded.append(w_padded)
    wav = torch.stack(wav_padded, dim=0).to(torch.float32)

    max_len_token = max([t.shape[0] for t in token])
    token_padded = []
    for t in token:
        t_padded = torch.zeros(max_len_token)
        t_padded[: t.shape[0]] = t
        token_padded.append(t_padded)
    token = torch.stack(token_padded, dim=0).to(torch.long)

    wav_len = torch.tensor(wav_len).to(torch.long)
    feature_len = torch.tensor(feature_len).to(torch.long)
    token_len = torch.tensor(token_len).to(torch.long)

    return (
        wav,
        wav_len,
        feature_len,
        token,
        token_len,
        speaker,
        filename,
    )
