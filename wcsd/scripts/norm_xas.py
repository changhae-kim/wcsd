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
        "../examples/Cu Ga XAS data/config.yaml" \
        "../examples/Cu Ga XAS data/index"

  By default, this looks for a `best.pt` checkpoint in the current directory. You can provide an alternative file path using the `--model` flag:

    python norm_xas.py \
        "../examples/Cu Ga XAS data/config.yaml" \
        "../examples/Cu Ga XAS data/index" \
        -m best.pt

  To normalize the raw spectra without denoising, remove the flags related to denoising:

    norm_xas \
        "../examples/Cu Ga XAS data/raw.yaml" \
        "../examples/Cu Ga XAS data/index"

  You may verify that `raw.yaml` is just `config.yaml` without the denoising flags.


"""

import argparse
import os
import tarfile
import yaml
from dacite import from_dict
from datetime import datetime

import numpy as np
import torch
from torch.nn import DataParallel, MSELoss

from larch.io.columnfile import read_ascii, write_ascii
from larch.xafs import autobk

from wcsd.model.net import WeightCentricSpectrumDenoiser
from wcsd.utils.config import Config

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description=(
            "Normalize XAS spectra from an index file, "
            "optionally denoise them with WCSD, "
            "and save analysis products."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("index", type=str, help="Path to the index file")
    parser.add_argument("-m", "--model", type=str, default="best.pt",
        help="Path to the WCSD checkpoint file (default=best.pt)")
    args = parser.parse_args()

    print("config", args.config)
    print("index", args.index)
    print("model", args.model)

    # Get configuration
    with open(args.config, "rt") as f:
        data = yaml.safe_load(f)
    config = from_dict(data_class=Config, data=data)

    print("io_channels", config.io_channels)
    print("mid_channels", config.mid_channels)
    print("Nt", config.Nt)
    print("Ns_list", config.Ns_list)
    print("num_slayers", config.num_slayers)
    print("spec_type", config.spec_type)
    print("denoise_channels", config.denoise_channels)
    print("labels", config.labels)
    print("pre_edge_kws", config.pre_edge_kws)

    device = torch.device(config.device)
    print("device", device)

    # Load data
    topdir = os.path.dirname(args.index)
    filepaths = []
    data_list = []
    batch = []
    d_list = []
    with open(args.index, "rt") as f:
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
                dat = read_ascii(filepath, labels=config.labels)
                batch.append(filepath)
                d_list.append(dat)
        if batch != []:
            filepaths.append(batch)
            data_list.append(d_list)

    # Set up model
    if config.denoise_channels is not None:
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
    pad = max(config.Ns_list) // 2
    x0, x1 = +pad, -pad
    score = 0.0
    for b, batch in enumerate(data_list):

        # No denoising
        if config.denoise_channels is None:
            loss = torch.tensor([0.0])
            for dat in batch:
                if config.spec_type == "transmission":
                    dat.mu = np.log(dat.i0/dat.it)
                else:
                    dat.mu = dat.iff/dat.i0

        # Denoising
        else:
            img = []
            for n, dat in enumerate(batch):
                img.append([])
                for channel in config.denoise_channels:
                    if channel in ["energy", "i0", "it", "iff"]:
                        img[-1].append(getattr(dat, channel))
                    elif channel == "mut":
                        img[-1].append(np.log(dat.i0/dat.it))
                    elif channel == "muff":
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
                for m, channel in enumerate(config.denoise_channels):
                    if channel in ["energy", "i0", "it", "iff"]:
                        setattr(dat, channel, dimg[0, m, n, x0:x1])
                if config.spec_type == "transmission":
                    if "mut" in config.denoise_channels:
                        m = config.denoise_channels.index("mut")
                        dat.mu = dimg[0, m, n, x0:x1]
                    else:
                        dat.mu = np.log(dat.i0 / dat.it)
                else:
                    if "muff" in config.denoise_channels:
                        m = config.denoise_channels.index("muff")
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
            autobk(dat.energy, dat.mu, group=dat, pre_edge_kws=config.pre_edge_kws)
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
