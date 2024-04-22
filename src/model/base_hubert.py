import numpy as np
import omegaconf
import torch
import torch.nn as nn

from src.model.components.conv_decoder import ConvDecoderHuBERT
from src.model.components.hubert_decoder import HuBERTDecoder
from src.model.utils import load_avhubert, load_raven, load_vatlm


class BaseHuBERTModel(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg["model"]["ssl_type"] == "avhubert":
            self.avhubert = load_avhubert(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
        elif cfg["model"]["ssl_type"] == "raven":
            self.raven = load_raven(cfg)
            hidden_channels = self.raven.attention_dim
        elif cfg["model"]["ssl_type"] == "vatlm":
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.vatlm.encoder_embed_dim
        elif cfg["model"]["ssl_type"] == "ensemble":
            self.avhubert = load_avhubert(cfg)
            self.raven = load_raven(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim
                + self.raven.attention_dim
                + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_avhubert_vatlm":
            self.avhubert = load_avhubert(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_avhubert_raven":
            self.avhubert = load_avhubert(cfg)
            self.raven = load_raven(cfg)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim + self.raven.attention_dim,
                hidden_channels,
            )
        elif cfg["model"]["ssl_type"] == "ensemble_raven_vatlm":
            self.raven = load_raven(cfg)
            self.vatlm = load_vatlm(cfg)
            hidden_channels = self.vatlm.encoder_embed_dim
            self.dropout = nn.Dropout(cfg["model"]["ssl_feature_dropout"])
            self.fuse_layer = nn.Linear(
                self.raven.attention_dim + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )

        if cfg["model"]["spk_emb_layer"]["use"]:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg["model"]["spk_emb_layer"]["dim"],
                hidden_channels,
            )

        self.conv_decoder = ConvDecoderHuBERT(cfg, hidden_channels)
        self.hubert_decoder = HuBERTDecoder(
            cfg, cfg["model"]["decoder"]["hubert"]["encoder_output_dim"]
        )

    def extract_feature_avhubert(
        self,
        lip: torch.Tensor,
        padding_mask: torch.Tensor | None,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            padding_mask (Padded elements must be True.): (B, T)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        x = self.avhubert(
            video=lip,
            audio=audio,
            return_res_output=False,
            padding_mask=padding_mask,
        )  # (B, T, C)
        return x

    def extract_feature_raven(
        self,
        lip: torch.Tensor,
        padding_mask: torch.Tensor | None,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            padding_mask (Padded elements must be True.): (B, T)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        padding_mask = ~padding_mask  # True for unmasked positions
        padding_mask = padding_mask.unsqueeze(1)  # (B, 1, T)
        x, _ = self.raven(
            xs=lip,
            masks=padding_mask,
        )  # (B, T, C)
        return x

    def extract_feature_vatlm(
        self,
        lip: torch.Tensor,
        padding_mask: torch.Tensor | None,
        audio: None,
    ) -> torch.Tensor:
        """
        args:
            lip: (B, C, T, H, W)
            padding_mask (Padded elements must be True.): (B, T)
            audio: (B, C, T)
        return:
            x: (B, T, C)
        """
        x = self.vatlm(
            video=lip,
            audio=audio,
            padding_mask=padding_mask,
        )  # (B, T, C)
        return x

    def compute_mask_indices(
        self,
        shape: tuple[int, int],
        padding_mask: torch.Tensor | None,
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
        require_same_masks: bool = True,
        mask_dropout: float = 0.0,
        add_masks: bool = False,
        seed: int | None = None,
        epoch: int | None = None,
        indices: torch.Tensor | None = None,
        idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
        num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
    ) -> np.ndarray:
        """
        Computes random mask spans for a given shape

        Args:
            shape: the the shape for which to compute masks.
                should be of size 2 where first element is batch size and 2nd is timesteps
            padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements.
                If True, the element will be ignored.
            mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
                number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
                however due to overlaps, the actual number will be smaller (unless no_overlap is True)
            mask_type: how to compute mask lengths
                static = fixed size
                uniform = sample from uniform distribution [mask_other, mask_length*2]
                normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
                poisson = sample from possion distribution with lambda = mask length
            min_masks: minimum number of masked spans
            no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
            min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
            require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
            mask_dropout: randomly dropout this percentage of masks in each example
        """

        bsz, all_sz = shape
        mask = np.full((bsz, all_sz), False)

        if num_mask_ver == 1:
            all_num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * all_sz / float(mask_length) + np.random.rand()
            )
            all_num_mask = max(min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):
            if seed is not None and epoch is not None and indices is not None:
                seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
            else:
                seed_i = None

            rng = np.random.default_rng(seed_i)

            if padding_mask is not None:
                sz = all_sz - padding_mask[i].long().sum().item()
                assert sz >= 0, sz
            else:
                sz = all_sz

            if num_mask_ver == 1:
                if padding_mask is not None:
                    num_mask = int(
                        # add a random number for probabilistic rounding
                        mask_prob * sz / float(mask_length) + np.random.rand()
                    )
                    num_mask = max(min_masks, num_mask)
                else:
                    num_mask = all_num_mask
            elif num_mask_ver == 2:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length) + rng.random()
                )
                num_mask = max(min_masks, num_mask)
            else:
                raise ValueError()

            if mask_type == "static":
                lengths = np.full(num_mask, mask_length)
            elif mask_type == "uniform":
                lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
            elif mask_type == "normal":
                lengths = rng.normal(mask_length, mask_other, size=num_mask)
                lengths = [max(1, int(round(x))) for x in lengths]
            elif mask_type == "poisson":
                lengths = rng.poisson(mask_length, size=num_mask)
                lengths = [int(round(x)) for x in lengths]
            else:
                raise Exception("unknown mask selection " + mask_type)

            if sum(lengths) == 0:
                if mask_type == "static":
                    raise ValueError("this should never happens")
                else:
                    lengths = [min(mask_length, sz - 1)]

            if no_overlap:
                mask_idc = []

                def arrange(s, e, length, keep_length):
                    span_start = rng.randint(s, e - length)
                    mask_idc.extend(span_start + i for i in range(length))

                    new_parts = []
                    if span_start - s - min_space >= keep_length:
                        new_parts.append((s, span_start - min_space + 1))
                    if e - span_start - length - min_space > keep_length:
                        new_parts.append((span_start + length + min_space, e))
                    return new_parts

                parts = [(0, sz)]
                min_length = min(lengths)
                for length in sorted(lengths, reverse=True):
                    lens = np.fromiter(
                        (e - s if e - s >= length + min_space else 0 for s, e in parts),
                        np.int,
                    )
                    l_sum = np.sum(lens)
                    if l_sum == 0:
                        break
                    probs = lens / np.sum(lens)
                    c = rng.choice(len(parts), p=probs)
                    s, e = parts.pop(c)
                    parts.extend(arrange(s, e, length, min_length))
                mask_idc = np.asarray(mask_idc)
            else:
                if idc_select_ver == 1:
                    min_len = min(lengths)
                    if sz - min_len <= num_mask:
                        min_len = sz - num_mask - 1
                    mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
                elif idc_select_ver == 2:
                    mask_idc = rng.choice(sz, num_mask, replace=False)
                else:
                    raise ValueError()

                mask_idc = np.asarray(
                    [
                        mask_idc[j] + offset
                        for j in range(len(mask_idc))
                        for offset in range(lengths[j])
                    ]
                )

            mask_idc = np.unique(mask_idc[mask_idc < sz])
            if len(mask_idc) >= sz:
                raise ValueError(
                    (
                        f"the entire sequence is masked. "
                        f"sz={sz}; mask_idc[mask_idc]; "
                        f"index={indices[i] if indices is not None else None}"
                    )
                )
            mask_idcs.append(mask_idc)

        target_len = None
        if require_same_masks:
            if add_masks:
                target_len = max([len(m) for m in mask_idcs])
            else:
                target_len = min([len(m) for m in mask_idcs])

        for i, mask_idc in enumerate(mask_idcs):
            if target_len is not None and len(mask_idc) > target_len:
                mask_idc = rng.choice(mask_idc, target_len, replace=False)

            mask[i, mask_idc] = True

            if target_len is not None and len(mask_idc) < target_len:
                unmasked = np.flatnonzero(~mask[i])
                to_mask = rng.choice(
                    unmasked, target_len - len(mask_idc), replace=False
                )
                mask[i, to_mask] = True

            if mask_dropout > 0:
                masked = np.flatnonzero(mask[i])
                num_holes = np.rint(len(masked) * mask_dropout).astype(int)
                to_drop = rng.choice(masked, num_holes, replace=False)
                mask[i, to_drop] = False

        return mask

    def forward(
        self,
        lip: torch.Tensor,
        audio: None,
        spk_emb: torch.Tensor,
        feature_hubert_prj: torch.Tensor | None,
        padding_mask_lip: torch.Tensor | None,
        padding_mask_feature_hubert: torch.Tensor | None,
    ) -> tuple:
        """
        args:
            lip: (B, C, H, W, T)
            audio: (B, C, T)
            lip_len: (B,)
            spk_emb: (B, C)
        return:
            output: (B, C, T)
        """
        lip = lip.permute(0, 1, 4, 2, 3)  # (B, C, T, H, W)

        if self.cfg["model"]["ssl_type"] == "avhubert":
            feature = self.extract_feature_avhubert(lip, padding_mask_lip, audio)
        elif self.cfg["model"]["ssl_type"] == "raven":
            feature = self.extract_feature_raven(lip, padding_mask_lip, audio)
        elif self.cfg["model"]["ssl_type"] == "vatlm":
            feature = self.extract_feature_vatlm(lip, padding_mask_lip, audio)
        elif self.cfg["model"]["ssl_type"] == "ensemble":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(
                    lip, padding_mask_lip, audio
                )
                feature_raven = self.extract_feature_raven(lip, padding_mask_lip, audio)
                feature_vatlm = self.extract_feature_vatlm(lip, padding_mask_lip, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_raven = self.dropout(feature_raven)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat(
                [feature_avhubert, feature_raven, feature_vatlm], dim=-1
            )
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_avhubert_vatlm":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(
                    lip, padding_mask_lip, audio
                )
                feature_vatlm = self.extract_feature_vatlm(lip, padding_mask_lip, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat([feature_avhubert, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_avhubert_raven":
            with torch.no_grad():
                feature_avhubert = self.extract_feature_avhubert(
                    lip, padding_mask_lip, audio
                )
                feature_raven = self.extract_feature_raven(lip, padding_mask_lip, audio)
            feature_avhubert = self.dropout(feature_avhubert)
            feature_raven = self.dropout(feature_raven)
            feature = torch.concat([feature_avhubert, feature_raven], dim=-1)
            feature = self.fuse_layer(feature)
        elif self.cfg["model"]["ssl_type"] == "ensemble_raven_vatlm":
            with torch.no_grad():
                feature_raven = self.extract_feature_raven(lip, padding_mask_lip, audio)
                feature_vatlm = self.extract_feature_vatlm(lip, padding_mask_lip, audio)
            feature_raven = self.dropout(feature_raven)
            feature_vatlm = self.dropout(feature_vatlm)
            feature = torch.concat([feature_raven, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)

        if self.cfg["model"]["spk_emb_layer"]["use"]:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)  # (B, T, C)
            feature = torch.cat([feature, spk_emb], dim=-1)
            feature = self.spk_emb_layer(feature)

        (
            conv_output_mel,
            conv_output_hubert_encoder,
            conv_output_hubert_cluster,
            conv_output_hubert_prj,
        ) = self.conv_decoder(feature)

        hubert_decoder_input = conv_output_hubert_prj
        mask_indices = torch.ones(
            feature_hubert_prj.shape[0], feature_hubert_prj.shape[2]
        ).to(dtype=torch.bool, device=lip.device)

        if self.cfg["model"]["decoder"]["hubert"]["encoder_input_mask"]["use"]:
            mask_indices = torch.from_numpy(
                self.compute_mask_indices(
                    shape=(feature_hubert_prj.shape[0], feature_hubert_prj.shape[2]),
                    padding_mask=padding_mask_feature_hubert,
                    mask_prob=self.cfg["model"]["decoder"]["hubert"][
                        "encoder_input_mask"
                    ]["mask_prob"],
                    mask_length=self.cfg["model"]["decoder"]["hubert"][
                        "encoder_input_mask"
                    ]["mask_length"],
                    mask_type="static",
                    min_masks=2,
                    no_overlap=False,
                    min_space=1,
                )
            ).to(dtype=torch.bool, device=feature_hubert_prj.device)  # (B, T)
            mask_indices_prj = mask_indices.unsqueeze(1).expand(
                -1, feature_hubert_prj.shape[1], -1
            )
            hubert_decoder_input = feature_hubert_prj.to(
                dtype=conv_output_hubert_prj.dtype
            )
            hubert_decoder_input[mask_indices_prj] = conv_output_hubert_prj[
                mask_indices_prj
            ]

        hubert_output_reg, hubert_output_cls = self.hubert_decoder(
            hubert_decoder_input.permute(0, 2, 1)
        )

        return (
            conv_output_mel,
            conv_output_hubert_prj,
            conv_output_hubert_encoder,
            conv_output_hubert_cluster,
            hubert_output_reg,
            hubert_output_cls,
            mask_indices,
        )
