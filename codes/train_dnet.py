import argparse
import shutil
import numpy as np
import torch

from datetime import datetime
from torch.nn import DataParallel, MSELoss
from torch.optim import NAdam, RAdam, Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dudvd_ensemble import MultiScaleBlindDenoiser

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath',     type=str,   nargs=None, required=True,     help='Path to data file')
    parser.add_argument('-i', '--io_channels',  type=int,   nargs='+',  required=True,     help='Indices of input/output channels')
    parser.add_argument(      '--mid_channels', type=int,   nargs=None, default=8,         help='Number of intermediate channels')
    parser.add_argument(      '--Nt',           type=int,   nargs=None, default=3,         help='Kernel size in temporal dimension')
    parser.add_argument(      '--Ns',           type=int,   nargs='+',  default=[3,5,7],   help='Kernel sizes in spatial dimension')
    parser.add_argument(      '--Dt',           type=int,   nargs=None, default=1,         help='Blind-spot size in temporal dimension')
    parser.add_argument(      '--Ds',           type=int,   nargs=None, default=1,         help='Blind-spot size in spatial dimension')
    parser.add_argument(      '--num_slayers',  type=int,   nargs=None, default=1,         help='Number of separable layers')
    parser.add_argument(      '--device',       type=str,   nargs=None, default='cuda',    help='Device')
    parser.add_argument('-x', '--axis',         type=int,   nargs=None, default=None,      help='Index of \"axis\" channel to be excluded in training (default=None)')
    parser.add_argument('-w', '--window',       type=int,   nargs='+',  default=None,      help='Limits on spatial indices to be used in training (default=None based on kernel size)')
    parser.add_argument(      '--opt',          type=str,   nargs=None, default='Adam',    help='Optimizer. Choose one of NAdam, RAdam, Adam, AdamW, or SGD (default=Adam)')
    parser.add_argument(      '--lr',           type=float, nargs=None, default=0.001,     help='Learning rate')
    parser.add_argument(      '--norm',         action='store_true',                       help='Normalize data')
    parser.add_argument(      '--nostd',        action='store_true',                       help='Do NOT standardize data')

    cmd_args = parser.parse_args()
    print('filepath',     cmd_args.filepath)
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
    print('opt',          cmd_args.opt)
    print('lr',           cmd_args.lr)
    print('norm',         cmd_args.norm)
    print('nostd',        cmd_args.nostd)

    device = torch.device(cmd_args.device)
    print('device', device)

    data = np.load(cmd_args.filepath)
    if cmd_args.filepath.endswith('.npz'):
        data = [v for _, v in data.items()]
    train_data = [torch.from_numpy(v[:,cmd_args.io_channels,:,:]).to(torch.float32).to(device) for v in data]
    del data
    if cmd_args.norm:
        train_shift = [v.min(-1, keepdim=True)[0] for v in train_data]
        train_scale = [v.max(-1, keepdim=True)[0] - v_shift for v, v_shift in zip(train_data, train_shift)]
        train_data = [(v - v_shift) / (v_scale + (v_scale == 0.0)) for v, v_shift, v_scale in zip(train_data, train_shift, train_scale)]
    elif not cmd_args.nostd:
        train_shift = [v.mean(-1, keepdim=True) for v in train_data]
        train_scale = [v.std(-1, keepdim=True) for v in train_data]
        train_data = [(v - v_shift) / (v_scale + (v_scale == 0.0)) for v, v_shift, v_scale in zip(train_data, train_shift, train_scale)]
    print('train_data', len(train_data))
    print(*[list(v.size()) for v in train_data])

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
    if torch.cuda.device_count() > 1:
        denoise_net = DataParallel(denoise_net)
        net = denoise_net.module
    else:
        net = denoise_net
    denoise_net.to(device)
    opt_cls_dict = {'NAdam': NAdam, 'RAdam': RAdam, 'Adam': Adam, 'AdamW': AdamW, 'SGD': SGD}
    opt_cls = opt_cls_dict[cmd_args.opt]
    optimizer = opt_cls(denoise_net.parameters(), lr=cmd_args.lr)
    loss_func = MSELoss()

    print('Model training started at', datetime.now())
    pad = max(cmd_args.Ns) // 2
    x0, x1 = +pad, -pad
    if cmd_args.window is not None:
        x0, x1 = cmd_args.window
    best_epoch = 0
    best_score = 1E99
    for epoch in range(300):

        denoise_net.train()
        for img in train_data:
            if np.random.random() > 0.5:
                img = torch.flip(img, (-1,))
            if np.random.random() > 0.5:
                img = -img
            optimizer.zero_grad()
            dimg = denoise_net(img)
            loss = loss_func(dimg[:,:,:,x0:x1], img[:,:,:,x0:x1])
            loss.backward()
            optimizer.step()
            del img, dimg, loss

        total_loss = 0.0
        denoise_net.eval()
        for img in train_data:
            dimg = denoise_net(img)
            loss = loss_func(dimg[:,:,:,x0:x1], img[:,:,:,x0:x1])
            total_loss += loss.item()
            del img, dimg, loss

        with open('latest.tar', 'wb') as f:
            torch.save(net.state_dict(), f)
        if total_loss < best_score:
            shutil.copy('latest.tar', 'best.tar')
            best_epoch = epoch
            best_score = total_loss

        print('epoch', epoch, total_loss)

    shutil.move('latest.tar', 'final.tar')
    print('best epoch', best_epoch, best_score)
    print('Model training finished at', datetime.now())

    return

if __name__ == '__main__':
    exit(main())
