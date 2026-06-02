import numpy as np
import torch

from torch import nn
from torch.nn.functional import softmax

class MultiScaleBlindDenoiser(nn.Module):
    """
    Take three self-blind CNNs with different kernel sizes, combine
    their outputs with a learnable convex combination, and compare
    that mixture to the noisy input through MSE.
    """
    def __init__(
        self,
        inp_channels    = 1,
        mid_channels    = 8,
        out_channels    = 1,
        Nt              = 3,           # temporal kernel size
        Ns_list         = [3, 5, 7],   # spatial kernel sizes
        Dt              = 1,           # temporal blind-spot size
        Ds              = 1,           # spatial blind-spot size
        num_slayers     = 1,
    ):
        super().__init__()

        # Three independent back-bones (weights *not* shared)
        self.subnets = nn.ModuleList([
            BlindUNet(
                inp_channels=inp_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                Nt=Nt, Ns=ks,
                Dt=Dt, Ds=Ds,
                num_slayers=num_slayers,
                )
            for ks in Ns_list
        ])

        # Learnable mixing weights (initialised to 1/len(Ns_list))
        init_w = torch.ones(len(Ns_list), dtype=torch.float32) / len(Ns_list)
        self.mix_weights = nn.Parameter(init_w)

    def forward(self, x):
        """
        x: (B, C, T, X)

        Returns
        -------
        combined : (B, C, T, X)
            Weighted sum of the three subnet outputs where
        """

        # Individual predictions – we only use the main 3-D output
        preds = [net(x) for net in self.subnets]        # list of (B,C,T',X)
        preds = torch.stack(preds, dim=0)               # (K,B,C,T',X)

        # Positive weights that sum to 1
        weights = softmax(self.mix_weights, dim=0)      # (K,)
        weights = weights.view(-1, 1, 1, 1, 1)          # broadcast to preds

        combined = (weights * preds).sum(dim=0)         # (B,C,T',X)

        return combined
    
def special_padding_2d(tensor, padding):
    # tensor shape: (batch, channels, depth, width)
    # If tensor.size(dim) > 1, then reflect
    # If tensor.size(dim) == 1, then replicate
    d_pad, w_pad = padding

    # Pad along depth (front and back)
    if d_pad > 0:
        if tensor.size(2) > 1:
            while tensor.size(2) < d_pad + 1:
                fpad = torch.flip(tensor[:,:,1:,:], dims=[2])
                bpad = torch.flip(tensor[:,:,:-1,:], dims=[2])
                d_pad -= tensor.size(2) - 1
                tensor = torch.cat([fpad, tensor, bpad], dim=2)
            fpad = torch.flip(tensor[:,:,1:1+d_pad,:], dims=[2])
            bpad = torch.flip(tensor[:,:,-d_pad-1:-1,:], dims=[2])
            tensor = torch.cat([fpad, tensor, bpad], dim=2)
        else:
            tensor = tensor.repeat(1,1,2*d_pad+1,1)

    # Pad along width (left and right)
    if w_pad > 0:
        if tensor.size(3) > 1:
            while tensor.size(3) < w_pad + 1:
                lpad = torch.flip(tensor[:,:,:,1:], dims=[3])
                rpad = torch.flip(tensor[:,:,:,:-1], dims=[3])
                w_pad -= tensor.size(3) - 1
                tensor = torch.cat([lpad, tensor, rpad], dim=3)
            lpad = torch.flip(tensor[:,:,:,1:1+w_pad], dims=[3])
            rpad = torch.flip(tensor[:,:,:,-w_pad-1:-1], dims=[3])
            tensor = torch.cat([lpad, tensor, rpad], dim=3)
        else:
            tensor = tensor.repeat(1,1,1,2*w_pad+1)

    return tensor

class BlindConv2D(nn.Module):
    def __init__(
            self,
            inp_channels=1,
            out_channels=1,
            Nt=3, Ns=5,
            Dt=1, Ds=1,
            ):
        super(BlindConv2D, self).__init__()
        self.conv = nn.Conv2d(
            inp_channels,
            out_channels,
            kernel_size=(Nt,Ns),
            padding=0,
            bias=False
        )
        self.Nt = Nt
        self.Ns = Ns
        self.Dt = Dt
        self.Ds = Ds
        nn.init.kaiming_normal_(self.conv.weight)
        Tc = Nt // 2
        Sc = Ns // 2
        Tw = Dt // 2
        Sw = Ds // 2
        with torch.no_grad():
            self.conv.weight[..., Tc-Tw:Tc+Tw+1, Sc-Sw:Sc+Sw+1] = 0.0
        self.conv.weight.register_hook(self._zero_central_grad)

    def _zero_central_grad(self, grad):
        Tc = self.Nt // 2
        Sc = self.Ns // 2
        Tw = self.Dt // 2
        Sw = self.Ds // 2
        grad[..., Tc-Tw:Tc+Tw+1, Sc-Sw:Sc+Sw+1] = 0.0
        return grad

    def forward(self, x):
        padding = (self.Nt//2, self.Ns//2)
        x = special_padding_2d(x, padding)
        return self.conv(x)

class BlindUNet(nn.Module):
    def __init__(
            self,
            inp_channels=1,
            mid_channels=1,
            out_channels=1,
            Nt=3, Ns=5,
            Dt=1, Ds=1,
            num_slayers=1,
            ):
        super(BlindUNet, self).__init__()
        layers = []

        # Define the input and output channels for each layer
        layers.append(BlindConv2D(
            inp_channels,
            mid_channels,
            Nt=Nt,Ns=Ns,
            Dt=Dt,Ds=Ds,
            ))
        for j in range(num_slayers):
            layers.append(nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=1,
                padding=0,
                bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        layers.append(nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False))

        # Use nn.Sequential to stack the layers
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
