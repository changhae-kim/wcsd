import os
import argparse
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import torch

from datetime import datetime
from larch.io.columnfile import read_ascii, write_ascii
from larch.xafs import autobk, xftf
from torch.nn import DataParallel, MSELoss

from dudvd_ensemble import MultiScaleBlindDenoiser

labels_dict = {
        '/sdcc/u/ckim6/BNL/Delafossite/db/XAS Cu Ga data/Cu Ga XAS data.idx': 'energy i0 it ir iff aux1 aux2 aux3 aux4',
        '/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.prebin.idx': 'energy  i0  it  ir  iff  aux1  aux2  aux3  aux4',
        }

erange_dict = {
        '/sdcc/u/ckim6/BNL/Delafossite/db/XAS Cu Ga data/Cu Ga XAS data.idx': (-np.inf, +np.inf),
        '/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.prebin.idx': (21997.000000, 22853.403311),
        }

pre_edge_kws = {
        '/sdcc/u/ckim6/BNL/Delafossite/db/XAS Cu Ga data/Cu Ga XAS data.idx': {
            'pre2': -150.0,
            'pre1':  -50.0,
            'nnorm': 4,
            },
        '/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.prebin.idx': {
            'nnorm': -1,
            },
        }

kmax_dict = {
        'Cu-K': 11,
        'Ga-K': 13,
        'Ru K': -1,
        }

def get_kmax(filepath):
    for k, v in kmax_dict.items():
        if k in filepath:
            return v

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath',     type=str,   nargs=None, required=True,     help='Path to data index')
    parser.add_argument('-s', '--spec',         type=str,   nargs=None, default='fluor',   help='XAS spec to analyze. Choose one of trans or fluor (default=fluor)')
    parser.add_argument('-t', '--targets',      type=str,   nargs='+',  default=['none'],  help='Target data to denoise. Choose none or N of i0, it, iff, mut, or muff (default=none)')
    parser.add_argument('-m', '--model',        type=str,   nargs=None, default=None,      help='Path to model. Required unless targets == [\'none\']')
    parser.add_argument('-i', '--io_channels',  type=int,   nargs='+',  default=None,      help='Indices of input/output channels. Required unless targets == [\'none\']')
    parser.add_argument(      '--mid_channels', type=int,   nargs=None, default=8,         help='Number of intermediate channels')
    parser.add_argument(      '--Nt',           type=int,   nargs=None, default=3,         help='Kernel size in temporal dimension')
    parser.add_argument(      '--Ns',           type=int,   nargs='+',  default=[3,5,7],   help='Kernel sizes in spatial dimension')
    parser.add_argument(      '--Dt',           type=int,   nargs=None, default=1,         help='Blind-spot size in temporal dimension')
    parser.add_argument(      '--Ds',           type=int,   nargs=None, default=1,         help='Blind-spot size in spatial dimension')
    parser.add_argument(      '--num_slayers',  type=int,   nargs=None, default=1,         help='Number of separable layers')
    parser.add_argument(      '--device',       type=str,   nargs=None, default='cuda',    help='Device')
    parser.add_argument('-x', '--axis',         type=int,   nargs=None, default=None,      help='Index of \"axis\" channel to be excluded in training (default=None)')
    parser.add_argument('-w', '--window',       type=int,   nargs='+',  default=None,      help='Limits on spatial indices to be used in training (default=None based on kernel size)')
    parser.add_argument(      '--trim',         type=int,   nargs='+',  default=None,      help='Limits on spatial indices to be saved as output (default=None copies window)')
    parser.add_argument(      '--norm',         action='store_true',                       help='Normalize data')
    parser.add_argument(      '--nostd',        action='store_true',                       help='Do NOT standardize data')

    cmd_args = parser.parse_args()
    print('filepath',     cmd_args.filepath)
    print('spec',         cmd_args.spec)
    print('targets',      cmd_args.targets)
    print('model',        cmd_args.model)
    print('io_channels',  cmd_args.io_channels)
    print('mid_channels', cmd_args.mid_channels)
    print('Nt',           cmd_args.Nt)
    print('Ns',           cmd_args.Ns)
    print('Dt',           cmd_args.Dt)
    print('Ds',           cmd_args.Ds)
    print('num_slayers',  cmd_args.num_slayers)
    print('device',       cmd_args.device)
    print('axis',         cmd_args.axis)
    print('window',       cmd_args.window)
    print('trim',         cmd_args.trim)
    print('norm',         cmd_args.norm)
    print('nostd',        cmd_args.nostd)

    device = torch.device(cmd_args.device)
    print('device', device)

    filepaths = []
    data_list = []
    batch = []
    d_list = []
    f = open(cmd_args.filepath, 'rt')
    emin, emax = erange_dict[cmd_args.filepath]
    for line in f:
        filepath = line.strip()
        if filepath.startswith('#'):
            continue
        elif filepath == '' and batch != []:
            filepaths.append(batch)
            data_list.append(d_list)
            batch = []
            d_list = []
        else:
            batch.append(filepath)
            dat = read_ascii(filepath, labels=labels_dict[cmd_args.filepath])
            key = np.array([i for i in np.argsort(dat.energy) if dat.energy[i] >= emin and dat.energy[i] <= emax], dtype=int)
            dat.energy = dat.energy[key]
            dat.i0     = dat.i0[key]
            dat.it     = dat.it[key]
            dat.iff    = dat.iff[key]
            d_list.append(dat)
    if batch != []:
        filepaths.append(batch)
        data_list.append(d_list)
    f.close()

    if cmd_args.targets[0] != 'none':
        io_channels = len(cmd_args.io_channels)
        print('io_channels', io_channels)
        denoise_net = MultiScaleBlindDenoiser(
                        inp_channels=io_channels,
                        mid_channels=cmd_args.mid_channels,
                        out_channels=io_channels,
                        Nt=cmd_args.Nt, Ns_list=cmd_args.Ns,
                        Dt=cmd_args.Dt, Ds=cmd_args.Ds,
                        num_slayers=cmd_args.num_slayers,
                        )
        with open(cmd_args.model, 'rb') as f:
            d = torch.load(f, weights_only=True, map_location='cpu')
            denoise_net.load_state_dict(d)
        if torch.cuda.device_count() > 1:
            denoise_net = DataParallel(denoise_net)
        denoise_net.to(device)
        denoise_net.eval()
        loss_func = MSELoss()

    print('Analysis started at', datetime.now())

    print('Denoising...')
    total_loss = 0.0
    pad = max(cmd_args.Ns) // 2
    x0, x1 = +pad, -pad
    if cmd_args.window is not None:
        x0, x1 = cmd_args.window
    dx0, dx1 = x0, x1
    if cmd_args.trim is not None:
        dx0, dx1 = cmd_args.trim
    for b, batch in enumerate(data_list):
        if cmd_args.targets == ['none']:
            loss = torch.tensor([0.0])
            for dat in batch:
                if cmd_args.spec == 'trans':
                    dat.mu = np.log(dat.i0/dat.it)
                elif cmd_args.spec == 'fluor':
                    dat.mu = dat.iff/dat.i0
        else:
            img = []
            for n, dat in enumerate(batch):
                img.append([])
                for target in cmd_args.targets:
                    if target in ['energy', 'i0', 'it', 'iff']:
                        img[-1].append(getattr(dat,target))
                    elif target == 'mut':
                        img[-1].append(np.log(dat.i0/dat.it))
                    elif target == 'muff':
                        img[-1].append(dat.iff/dat.i0)
            img = np.array(img).swapaxes(0,1)[None,...]
            img = torch.from_numpy(img).to(torch.float32).to(device)
            if cmd_args.norm:
                v_shift = img.min(-1, keepdim=True)[0]
                v_scale = img.max(-1, keepdim=True)[0] - v_shift
                img = (img - v_shift) / (v_scale + (v_scale == 0.0))
            elif not cmd_args.nostd:
                v_shift = img.mean(-1, keepdim=True)
                v_scale = img.std(-1, keepdim=True)
                img = (img - v_shift) / (v_scale + (v_scale == 0.0))
            dimg = denoise_net(img)
            loss = loss_func(dimg[:,:,:,x0:x1], img[:,:,:,x0:x1])
            print(filepaths[b][n], loss.item())

            if cmd_args.norm or not cmd_args.nostd:
                dimg = v_scale * dimg + v_shift
            dimg = dimg.clone().detach().cpu().numpy()
            for n, dat in enumerate(batch):
                dat.energy = dat.energy[dx0:dx1]
                dat.i0     = dat.i0[dx0:dx1]
                dat.it     = dat.it[dx0:dx1]
                dat.iff    = dat.iff[dx0:dx1]
                for m, target in enumerate(cmd_args.targets):
                    if target in ['energy', 'i0', 'it', 'iff']:
                        setattr(dat, target, dimg[0,m,n,dx0:dx1])
                if cmd_args.spec == 'trans':
                    if 'mut' in cmd_args.targets:
                        m = cmd_args.targets.index('mut')
                        dat.mu = dimg[0,m,n,dx0:dx1]
                    else:
                        dat.mu = np.log(dat.i0/dat.it)
                else:
                    if 'muff' in cmd_args.targets:
                        m = cmd_args.targets.index('muff')
                        dat.mu = dimg[0,m,n,dx0:dx1]
                    else:
                        dat.mu = dat.iff/dat.i0

        total_loss += loss.item()
    print('total loss', total_loss)

    print('Analyzing...')
    fignames = []
    datnames = []
    for b, batch in enumerate(data_list):
        for n, dat in enumerate(batch):

            figname = f'mu_{b}_{n}.png'
            plt.figure()
            plt.plot(dat.energy, dat.mu)
            plt.xlim(dat.energy.min(), dat.energy.max())
            plt.xlabel(r'$E$ (eV)')
            plt.ylabel(r'$\mu(E)$')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            fignames.append(figname)
            datname = f'mu_{b}_{n}.dat'
            write_ascii(datname, dat.energy, dat.mu)
            datnames.append(datname)

            autobk(dat.energy, dat.mu, group=dat, pre_edge_kws=pre_edge_kws[cmd_args.filepath])

            figname = f'flat_{b}_{n}.png'
            plt.figure()
            plt.plot(dat.energy, dat.flat)
            plt.xlim(dat.energy.min(), dat.energy.max())
            plt.xlabel(r'$E$ (eV)')
            plt.ylabel(r'$\mu(E)$')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            fignames.append(figname)
            datname = f'flat_{b}_{n}.dat'
            write_ascii(datname, dat.energy, dat.flat)
            datnames.append(datname)

            chi0 = dat.chi * dat.k**0
            chi1 = dat.chi * dat.k**1
            chi2 = dat.chi * dat.k**2
            chi0 = chi0 / np.abs(chi0).max()
            chi1 = chi1 / np.abs(chi1).max()
            chi2 = chi2 / np.abs(chi2).max()

            figname = f'chik_{b}_{n}.png'
            plt.figure()
            plt.plot(dat.k, chi0, label='$w = 0$')
            plt.plot(dat.k, chi1, label='$w = 1$')
            plt.plot(dat.k, chi2, label='$w = 2$')
            plt.xlim(dat.k.min(), dat.k.max())
            plt.xlabel(r'$k (\AA^{-1})$')
            plt.ylabel(r'Normalized $k^w \chi(k)$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            fignames.append(figname)
            datname = f'chik_{b}_{n}.dat'
            write_ascii(datname, dat.k, dat.chi)
            datnames.append(datname)

            kmax = get_kmax(filepaths[b][n])
            xftf(dat.k, dat.chi, group=dat, kweight=2, kmin=1, kmax=kmax, dk=2, window='hanning')

            figname = f'chir_{b}_{n}.png'
            plt.figure()
            plt.plot(dat.r, dat.chir_mag, 'k-', label=r'$|\chi(R)|$')
            plt.plot(dat.r, dat.chir_re,  'b-', label=r'$\Re[\chi(R)]$')
            plt.plot(dat.r, dat.chir_im,  'r-', label=r'$\Im[\chi(R)]$')
            plt.xlim(dat.r.min(), dat.r.max())
            plt.xlabel(r'$R (\AA)$')
            plt.ylabel(r'$\chi(R)$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            fignames.append(figname)
            datname = f'chir_{b}_{n}.dat'
            write_ascii(datname, dat.r, dat.chir_mag, dat.chir_re, dat.chir_im)
            datnames.append(datname)

    with tarfile.open('images.tgz', 'w:gz') as tar:
        for figname in fignames:
            tar.add(figname)
    for figname in fignames:
        os.remove(figname)
    with tarfile.open('data.tgz', 'w:gz') as tar:
        for datname in datnames:
            tar.add(datname)
    for datname in datnames:
        os.remove(datname)
    print('Analysis finished at', datetime.now())

    return

if __name__ == '__main__':
    exit(main())
