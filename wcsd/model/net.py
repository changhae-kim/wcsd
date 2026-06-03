"""
Weight-Centric Spectrum Denoiser (WCSD) model components.

This module defines:
- BlindConv2D: a convolutional layer with a blind spot at the center pixel.
- BlindUNet: a shallow blind CNN stack used as an ensemble member.
- WeightCentricSpectrumDenoiser: an ensemble of blind CNN backbones with
  learnable softmax mixing weights.

The model expects 4D inputs with shape (batch, channels, timesteps, energies).
"""

import torch
from torch import nn
from torch.nn.functional import pad, softmax

__all__ = ["BlindConv2D", "BlindUNet", "WeightCentricSpectrumDenoiser"]

def special_padding_2d(x, padding):
    """Pad a 4D tensor of shape (batch, channels, timesteps, energies)."""
    # For each dim, if x.size(dim) > 1, then reflect, else replicate.
    d_pad, w_pad = padding
    if d_pad > 0:
        if x.size(2) > 1:
            x = pad(x, (0, 0, d_pad, d_pad), mode="reflect")
        else:
            x = x.repeat(1, 1, 2 * d_pad + 1, 1)
    if w_pad > 0:
        if x.size(3) > 1:
            x = pad(x, (w_pad, w_pad, 0, 0), mode="reflect")
        else:
            x = x.repeat(1, 1, 1, 2 * w_pad + 1)
    return x

class BlindConv2D(nn.Module):
    """2D convolution with a blind spot at the center pixel."""

    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        Nt=3,
        Ns=5,
    ):
        super(BlindConv2D, self).__init__()

        # Create the convolutional kernel
        self.conv = nn.Conv2d(
            inp_channels,
            out_channels,
            kernel_size=(Nt,Ns),
            padding=0,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight)

        # Zero out the central blind spot
        Tc = Nt // 2
        Sc = Ns // 2
        with torch.no_grad():
            self.conv.weight[..., Tc:Tc+1, Sc:Sc+1] = 0.0

        # Register a gradient hook so the center stays zero in training
        self.conv.weight.register_hook(self._zero_central_grad)

        # Save the kernel dimensions for _zero_central_grad
        self.Nt = Nt
        self.Ns = Ns

        return

    def _zero_central_grad(self, grad):
        Tc = self.Nt // 2
        Sc = self.Ns // 2
        grad[..., Tc:Tc+1, Sc:Sc+1] = 0.0
        return grad

    def forward(self, x):
        padding = (self.Nt//2, self.Ns//2)
        x = special_padding_2d(x, padding)
        return self.conv(x)

class BlindUNet(nn.Module):
    """A shallow blind CNN stack used as a WCSD ensemble member."""

    def __init__(
        self,
        inp_channels=1,
        mid_channels=1,
        out_channels=1,
        Nt=3,
        Ns=5,
        num_slayers=1,
    ):
        super(BlindUNet, self).__init__()
        layers = []

        # Define the input and output channels for each layer
        layers.append(BlindConv2D(
            inp_channels,
            mid_channels,
            Nt=Nt,
            Ns=Ns,
        ))
        for j in range(num_slayers):
            layers.append(nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        layers.append(nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        ))

        # Use nn.Sequential to stack the layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class WeightCentricSpectrumDenoiser(nn.Module):
    """Ensemble of blind CNN backbones combined with learnable mixing weights."""

    def __init__(
        self,
        inp_channels = 1,
        mid_channels = 8,
        out_channels = 1,
        Nt           = 3,           # temporal kernel size
        Ns_list      = [3, 5, 7],   # spatial kernel sizes
        num_slayers  = 1,
    ):
        super(WeightCentricSpectrumDenoiser, self).__init__()

        # Three independent back-bones (weights *not* shared)
        self.subnets = nn.ModuleList([
            BlindUNet(
                inp_channels=inp_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                Nt=Nt,
                Ns=ks,
                num_slayers=num_slayers,
                )
            for ks in Ns_list
        ])

        # Learnable mixing weights (softmax applied in forward)
        init_w = torch.zeros(len(Ns_list), dtype=torch.float32)
        self.mix_weights = nn.Parameter(init_w)

    def forward(self, x):
        """Return the softmax-weighted mixture of the subnet predictions."""

        # Individual predictions – we only use the main 3-D output
        preds = [net(x) for net in self.subnets]        # list of (B,C,T',X)
        preds = torch.stack(preds, dim=0)               # (K,B,C,T',X)

        # Positive weights that sum to 1
        weights = softmax(self.mix_weights, dim=0)      # (K,)
        weights = weights.view(-1, 1, 1, 1, 1)          # broadcast to preds

        combined = (weights * preds).sum(dim=0)         # (B,C,T',X)

        return combined
