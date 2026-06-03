"""
Normalize XAS spectra with optional WCSD denoising and save the results.

Workflow
--------
1. Read an index file containing one spectrum path per line.
2. Optionally denoise selected channels with a WCSD model.
3. Run background subtraction with Larch.
4. Save per-spectrum ASCII outputs and compressed archives.

Example
-------

    python norm_xas.py \
        -f "../examples/Cu Ga XAS data/index" \
        -c "../examples/Cu Ga XAS data/config.json" \
        -t muff \
        -m best.pt \
        -i 2
"""

import argparse
import json
import os
import tarfile
from datetime import datetime

import numpy as np
import torch
from torch.nn import DataParallel, MSELoss

from larch.io.columnfile import read_ascii, write_ascii
from larch.xafs import autobk

from wcsd.model.net import WeightCentricSpectrumDenoiser

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Normalize XAS spectra from an index file, optionally denoise them with WCSD, and save analysis products.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to the index file")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to the config .json file")
    parser.add_argument("-s", "--spec", type=str, default="fluor", choices=("trans", "fluor"), help="XAS spectrum type to analyze (default=fluor)")
    parser.add_argument("-t", "--targets", type=str, nargs="+", default=["none"], help="Target channels to denoise. Choose none or N of i0, it, iff, mut, or muff (default=none)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the WCSD checkpoint file (required unless targets == [\"none\"])")
    parser.add_argument("-i", "--io_channels", type=int, nargs="+", default=None, help="Indices of channels used as both input and output (required unless targets == [\"none\"])")
    parser.add_argument("--mid_channels", type=int, default=8, help="Number of intermediate channels (default=8)")
    parser.add_argument("--Nt", type=int, default=3, help="Kernel size in temporal dimension (default=3)")
    parser.add_argument("--Ns", type=int, nargs="+", default=[3, 5, 7], help="Kernel sizes in spectral dimension (default=[3, 5, 7])")
    parser.add_argument("--num_slayers", type=int, default=1, help="Number of separable layers in each subnet (default=1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default=cuda)")

    args = parser.parse_args()
    print("filepath", args.filepath)
    print("config", args.config)
    print("spec", args.spec)
    print("targets", args.targets)
    print("model", args.model)
    print("io_channels", args.io_channels)
    print("mid_channels", args.mid_channels)
    print("Nt", args.Nt)
    print("Ns", args.Ns)
    print("num_slayers", args.num_slayers)

    device = torch.device(args.device)
    print("device", device)

    # Get dataset configurations
    with open(args.config, "rt") as f:
        config = json.load(f)
    labels = config["labels"]
    pre_edge_kws = config["pre_edge_kws"]

    # Load data
    topdir = os.path.dirname(args.filepath)
    filepaths = []
    data_list = []
    batch = []
    d_list = []
    with open(args.filepath, "rt") as f:
        for line in f:
            filepath = line.strip()
            if filepath.startswith("#"):
                continue
            elif filepath == "" and batch != []:
                filepaths.append(batch)
                data_list.append(d_list)
                batch = []
                d_list = []
            else:
                filepath = os.path.join(topdir, filepath)
                dat = read_ascii(filepath, labels=labels)
                batch.append(filepath)
                d_list.append(dat)
        if batch != []:
            filepaths.append(batch)
            data_list.append(d_list)

    # Set up model
    if args.targets[0] != "none":
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
        with open(args.model, "rb") as f:
            d = torch.load(f, weights_only=True, map_location="cpu")
            denoise_net.load_state_dict(d)
        if torch.cuda.device_count() > 1:
            denoise_net = DataParallel(denoise_net)
        denoise_net.to(device)
        denoise_net.eval()
        loss_func = MSELoss()

    print("Analysis started at", datetime.now())

    print("Denoising...")
    pad = max(args.Ns) // 2
    x0, x1 = +pad, -pad
    score = 0.0
    for b, batch in enumerate(data_list):

        # No denoising
        if args.targets == ["none"]:
            loss = torch.tensor([0.0])
            for dat in batch:
                if args.spec == "trans":
                    dat.mu = np.log(dat.i0/dat.it)
                elif args.spec == "fluor":
                    dat.mu = dat.iff/dat.i0

        # Denoising
        else:
            img = []
            for n, dat in enumerate(batch):
                img.append([])
                for target in args.targets:
                    if target in ["energy", "i0", "it", "iff"]:
                        img[-1].append(getattr(dat, target))
                    elif target == "mut":
                        img[-1].append(np.log(dat.i0/dat.it))
                    elif target == "muff":
                        img[-1].append(dat.iff/dat.i0)
            img = np.array(img).swapaxes(0, 1)[None, ...]
            img = torch.from_numpy(img).to(torch.float32).to(device)

            # Standardize batches
            v_shift = img.mean(-1, keepdim=True)
            v_scale = img.std(-1, keepdim=True)
            img = (img - v_shift) / (v_scale + (v_scale == 0.0))

            # Denoise
            dimg = denoise_net(img)
            loss = loss_func(dimg[:, :, :, x0:x1], img[:, :, :, x0:x1])
            print(filepaths[b][n], loss.item())

            # Rescale batches
            dimg = v_scale * dimg + v_shift
            dimg = dimg.clone().detach().cpu().numpy()

            # Put results into data stores
            for n, dat in enumerate(batch):
                dat.energy = dat.energy[x0:x1]
                dat.i0 = dat.i0[x0:x1]
                dat.it = dat.it[x0:x1]
                dat.iff = dat.iff[x0:x1]
                for m, target in enumerate(args.targets):
                    if target in ["energy", "i0", "it", "iff"]:
                        setattr(dat, target, dimg[0, m, n, x0:x1])
                if args.spec == "trans":
                    if "mut" in args.targets:
                        m = args.targets.index("mut")
                        dat.mu = dimg[0, m, n, x0:x1]
                    else:
                        dat.mu = np.log(dat.i0 / dat.it)
                else:
                    if "muff" in args.targets:
                        m = args.targets.index("muff")
                        dat.mu = dimg[0, m, n, x0:x1]
                    else:
                        dat.mu = dat.iff / dat.i0

        score += loss.item()
    print("score", score)

    # Normalize XAS spectra and save results
    print("Analyzing...")
    filenames = []
    for b, batch in enumerate(data_list):
        for n, dat in enumerate(batch):
            autobk(dat.energy, dat.mu, group=dat, pre_edge_kws=pre_edge_kws)
            filename = f"flat_{b}_{n}.dat"
            write_ascii(filename, dat.energy, dat.flat, label="energy flat")
            filenames.append(filename)
    with tarfile.open("flat.tgz", "w:gz") as tar:
        for filename in filenames:
            tar.add(filename)
    for filename in filenames:
        os.remove(filename)
    print("Analysis finished at", datetime.now())

    return

if __name__ == "__main__":
    exit(main())
