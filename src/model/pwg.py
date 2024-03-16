import logging
import math

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleNet(nn.Module):
    def __init__(self, upsample_scales):
        super().__init__()
        self.upsample_scales = upsample_scales

        convs = []
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            padding = (0, (kernel_size[1] - 1) // 2)
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        1, 1, kernel_size=kernel_size, padding=padding, bias=False
                    ),
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.unsqueeze(1)
        for scale, layer in zip(self.upsample_scales, self.convs):
            x = F.interpolate(x, scale_factor=(1, scale), mode="nearest")
            x = layer(x)
        return x.squeeze(1)


class ConvinUpSampleNet(nn.Module):
    def __init__(self, in_channels, upsample_scales):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm1d(in_channels),
            # nn.ReLU(),
        )
        self.upsample_layers = UpSampleNet(upsample_scales)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.upsample_layers(x)
        return x


class WaveNetResBlock(nn.Module):
    def __init__(self, inner_channels, cond_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            inner_channels,
            int(inner_channels * 2),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.cond_layer = nn.Conv1d(
            cond_channels, int(inner_channels * 2), kernel_size=1
        )
        self.out_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.skip_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(inner_channels)

    def forward(self, x, c):
        """
        x, c : (B, C, T)
        """
        out = self.dropout(x)
        out = self.conv(out)
        out1, out2 = torch.split(out, out.shape[1] // 2, dim=1)

        c = self.cond_layer(c)
        c1, c2 = torch.split(c, c.shape[1] // 2, dim=1)
        out1 = out1 + c1
        out2 = out2 + c2
        out = torch.tanh(out1) * torch.sigmoid(out2)

        skip_out = self.skip_layer(out)
        out = (self.out_layer(out) + x) * math.sqrt(0.5)
        # out = self.bn(out)

        return out, skip_out


class Generator(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg["model"]["pwg"]["generator"]["in_channels"]
        out_channels = cfg["model"]["pwg"]["generator"]["out_channels"]
        inner_channels = cfg["model"]["pwg"]["generator"]["hidden_channels"]
        cond_channels = cfg["model"]["pwg"]["generator"]["cond_channels"]
        upsample_scales = cfg["model"]["pwg"]["generator"]["upsample_scales"]
        n_layers = cfg["model"]["pwg"]["generator"]["n_layers"]
        n_stacks = cfg["model"]["pwg"]["generator"]["n_stacks"]
        dropout = cfg["model"]["pwg"]["generator"]["dropout"]
        kernel_size = cfg["model"]["pwg"]["generator"]["kernel_size"]
        use_weight_norm = cfg["model"]["pwg"]["generator"]["use_weight_norm"]

        layers_per_stack = n_layers // n_stacks
        self.first_conv = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        self.cond_upsample_layers = ConvinUpSampleNet(cond_channels, upsample_scales)

        convs = []
        for i in range(n_layers):
            dilation = 2 ** (i % layers_per_stack)
            convs.append(
                WaveNetResBlock(
                    inner_channels=inner_channels,
                    cond_channels=cond_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Conv1d(inner_channels, inner_channels, kernel_size=1),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Conv1d(inner_channels, out_channels, kernel_size=1),
        )

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x, c : (B, C, T)
        """
        c = self.cond_upsample_layers(c)

        x = self.first_conv(x)
        skips = 0
        for layer in self.convs:
            x, skip = layer(x, c)
            skips += skip
        skips *= math.sqrt(1.0 / len(self.convs))

        x = skips
        x = self.out_layers(x)

        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class Discriminator(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg["model"]["pwg"]["discriminator"]["in_channels"]
        out_channels = cfg["model"]["pwg"]["discriminator"]["out_channels"]
        inner_channels = cfg["model"]["pwg"]["discriminator"]["hidden_channels"]
        n_layers = cfg["model"]["pwg"]["discriminator"]["n_layers"]
        kernel_size = cfg["model"]["pwg"]["discriminator"]["kernel_size"]
        use_weight_norm = cfg["model"]["pwg"]["discriminator"]["use_weight_norm"]
        dropout = cfg["model"]["pwg"]["discriminator"]["dropout"]

        convs = []
        for i in range(n_layers - 1):
            if i == 0:
                dilation = 1
                conv_in_channels = in_channels
            else:
                dilation = i
                conv_in_channels = inner_channels

            padding = (kernel_size - 1) // 2 * dilation
            convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        conv_in_channels,
                        inner_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                )
            )
        convs.append(
            nn.Conv1d(
                inner_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        )
        self.convs = nn.ModuleList(convs)

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """
        x : (B, 1, T)
        """
        for layer in self.convs:
            x = layer(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
