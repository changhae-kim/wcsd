"""
Train the Weight-Centric Spectrum Denoiser (WCSD).

Input
-----
The training data is in an .npz archive produced by `gather_data.py`. Each
array is expected to have the shape (1, n_channels, n_spectra, n_points).

Example
-------

    python train_net.py -f data.npz -i 2
"""

import argparse
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.nn import DataParallel, MSELoss
from torch.optim import NAdam, RAdam, Adam, AdamW, SGD

from wcsd.model.net import WeightCentricSpectrumDenoiser

OPTIMIZERS = {
    "NAdam": NAdam,
    "RAdam": RAdam,
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the WCSD on a dataset prepared with `gather_data.py`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to the .npz data file")
    parser.add_argument("-i", "--io_channels", type=int, nargs="+", required=True, help="Indices of channels to be used as both the input and the output")
    parser.add_argument("--mid_channels", type=int, default=8, help="Number of intermediate channels (default=8)")
    parser.add_argument("--Nt", type=int, default=3, help="Kernel size in temporal dimension (default=3)")
    parser.add_argument("--Ns", type=int, nargs="+",  default=[3, 5, 7], help="Kernel sizes in spectral dimension (default=[3, 5, 7])")
    parser.add_argument("--num_slayers", type=int, default=1, help="Number of separable layers in each subnet (default=1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default=cuda)")
    parser.add_argument("--opt", type=str, default="Adam", choices=sorted(OPTIMIZERS), help="Optimizer (default=Adam)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default=1e-3)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs (default=300)")

    args = parser.parse_args()
    print("filepath", args.filepath)
    print("io_channels", args.io_channels)
    print("mid_channels", args.mid_channels)
    print("Nt", args.Nt)
    print("Ns", args.Ns)
    print("num_slayers", args.num_slayers)
    print("device", args.device)
    print("opt", args.opt)
    print("lr", args.lr)
    print("epochs", args.epochs)

    device = torch.device(args.device)
    print("device", device)

    # Load data
    data = np.load(args.filepath)
    data = [v for _, v in data.items()]
    train_data = [torch.from_numpy(v[:, args.io_channels, :, :]).to(torch.float32).to(device) for v in data]
    del data

    # Standardize batches
    train_shift = [v.mean(-1, keepdim=True) for v in train_data]
    train_scale = [v.std(-1, keepdim=True) for v in train_data]
    train_data = [(v - v_shift) / (v_scale + (v_scale == 0.0)) for v, v_shift, v_scale in zip(train_data, train_shift, train_scale)]
    print("train_data", len(train_data))
    print(*[list(v.size()) for v in train_data])

    # Set up model and optimizer
    io_channels = len(args.io_channels)
    print("io_channels", io_channels)
    denoise_net = WeightCentricSpectrumDenoiser(
        inp_channels=io_channels,
        mid_channels=args.mid_channels,
        out_channels=io_channels,
        Nt=args.Nt,
        Ns_list=args.Ns,
        num_slayers=args.num_slayers,
    )
    if torch.cuda.device_count() > 1:
        denoise_net = DataParallel(denoise_net)
        net = denoise_net.module
    else:
        net = denoise_net
    denoise_net.to(device)
    optimizer = OPTIMIZERS[args.opt](denoise_net.parameters(), lr=args.lr)
    loss_func = MSELoss()

    print("Model training started at", datetime.now())
    pad = max(args.Ns) // 2
    x0, x1 = +pad, -pad
    best_epoch = 0
    best_score = float("inf")
    for epoch in range(args.epochs):

        # Training step
        denoise_net.train()
        for img in train_data:
            if np.random.random() > 0.5:
                img = torch.flip(img, (-1,))
            if np.random.random() > 0.5:
                img = -img
            optimizer.zero_grad()
            dimg = denoise_net(img)
            loss = loss_func(dimg[:, :, :, x0:x1], img[:, :, :, x0:x1])
            loss.backward()
            optimizer.step()
            del img, dimg, loss

        # Evaluation step
        denoise_net.eval()
        score = 0.0
        for img in train_data:
            dimg = denoise_net(img)
            loss = loss_func(dimg[:, :, :, x0:x1], img[:, :, :, x0:x1])
            score += loss.item()
            del img, dimg, loss

        # Save model weights
        with open("latest.pt", "wb") as f:
            torch.save(net.state_dict(), f)

        if score < best_score:
            shutil.copy("latest.pt", "best.pt")
            best_epoch = epoch
            best_score = score

        print("epoch", epoch, score)

    shutil.move("latest.pt", "final.pt")
    print("best epoch", best_epoch, best_score)
    print("Model training finished at", datetime.now())

    return

if __name__ == "__main__":
    exit(main())
