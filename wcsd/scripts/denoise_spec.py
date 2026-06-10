"""
Denoize XAS spectra with WCSD and save the results.

Workflow
--------
1. Read an index file containing one spectrum path per line.
2. Denoise selected channels with a WCSD model.
3. Save an ASCII output per spectrum and compress into an archive.

Example
-------

    python denoise_spec.py \
        "../examples/Cu Ga XAS data/config.yaml" \
        "../examples/Cu Ga XAS data/index"

  By default, this looks for a `best.pt` checkpoint in the current directory. You can provide an alternative file path using the `--model` flag:

    python denoise_spec.py \
        "../examples/Cu Ga XAS data/config.yaml" \
        "../examples/Cu Ga XAS data/index" \
        -m best.pt
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

from wcsd.model.net import WeightCentricSpectrumDenoiser
from wcsd.utils.config import Config

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Denoise XAS spectra from an index file with WCSD and save results.",
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

    print("Denoising started at", datetime.now())
    pad = max(config.Ns_list) // 2
    x0, x1 = +pad, -pad
    score = 0.0
    for b, batch in enumerate(data_list):

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
                else:
                    raise ValueError("Invalid item in denoise_channels.")
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
            elif config.spec_type == "fluorescence":
                if "muff" in config.denoise_channels:
                    m = config.denoise_channels.index("muff")
                    dat.mu = dimg[0, m, n, x0:x1]
                else:
                    dat.mu = dat.iff / dat.i0
            else:
                raise ValueError("Invalid spec_type.")

        score += loss.item()
    print("score", score)
    print("Denoising finished at", datetime.now())

    # Save results
    print("Saving results...")
    filenames = []
    for b, batch in enumerate(data_list):
        for n, dat in enumerate(batch):
            filename = f"denoised_{b}_{n}.dat"
            write_ascii(filename, dat.energy, dat.mu, label="energy mu")
            filenames.append(filename)
    with tarfile.open("denoised.tgz", "w:gz") as tar:
        for filename in filenames:
            tar.add(filename)
    for filename in filenames:
        os.remove(filename)

    return

if __name__ == "__main__":
    exit(main())
