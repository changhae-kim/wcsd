"""
Train the Weight-Centric Spectrum Denoiser (WCSD).

Input
-----
The training data is in an .npz archive produced by `gather_data.py`. Each
array is expected to have the shape (1, n_channels, n_spectra, n_points).

Example
-------

    python train_net.py "../examples/Cu Ga XAS data/config.yaml"

  By default, this looks for a `data.npz` archive in the current directory. You can provide an alternative file path using the `--input` flag:

    python train_net.py "../examples/Cu Ga XAS data/config.yaml" -i data.npz

"""

import argparse
import shutil
import yaml
from dacite import from_dict
from datetime import datetime

import numpy as np
import torch
from torch.nn import DataParallel, MSELoss
from torch.optim import NAdam, RAdam, Adam, AdamW, SGD

from wcsd.model.net import WeightCentricSpectrumDenoiser
from wcsd.utils.config import Config

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
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("-d", "--dataset", type=str, default="data.npz",
        help="Path to the NPZ data file (default=data.npz)")
    args = parser.parse_args()

    print("config", args.config)
    print("dataset", args.dataset)

    # Get configuration
    with open(args.config, "rt") as f:
        data = yaml.safe_load(f)
    config = from_dict(data_class=Config, data=data)

    print("io_channels", config.io_channels)
    print("mid_channels", config.mid_channels)
    print("Nt", config.Nt)
    print("Ns_list", config.Ns_list)
    print("num_slayers", config.num_slayers)
    print("optimizer", config.optimizer)
    print("lr", config.lr)
    print("epochs", config.epochs)

    device = torch.device(config.device)
    print("device", device)

    # Load data
    data = np.load(args.dataset)
    data = [v for _, v in data.items()]
    train_data = [torch.from_numpy(v[:, config.io_channels, :, :]).to(torch.float32).to(device) for v in data]
    del data

    # Standardize batches
    train_shift = [v.mean(-1, keepdim=True) for v in train_data]
    train_scale = [v.std(-1, keepdim=True) for v in train_data]
    train_data = [(v - v_shift) / (v_scale + (v_scale == 0.0)) for v, v_shift, v_scale in zip(train_data, train_shift, train_scale)]
    print("train_data", len(train_data))
    print(*[list(v.size()) for v in train_data])

    # Set up model and optimizer
    io_channels = len(config.io_channels)
    print("io_channels", io_channels)
    denoise_net = WeightCentricSpectrumDenoiser(
        inp_channels=io_channels,
        mid_channels=config.mid_channels,
        out_channels=io_channels,
        Nt=config.Nt,
        Ns_list=config.Ns_list,
        num_slayers=config.num_slayers,
    )
    if torch.cuda.device_count() > 1:
        denoise_net = DataParallel(denoise_net)
        net = denoise_net.module
    else:
        net = denoise_net
    denoise_net.to(device)
    optimizer = OPTIMIZERS[config.optimizer](denoise_net.parameters(), lr=config.lr)
    loss_func = MSELoss()

    print("Model training started at", datetime.now())
    pad = max(config.Ns_list) // 2
    x0, x1 = +pad, -pad
    best_epoch = 0
    best_score = float("inf")
    for epoch in range(config.epochs):

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
