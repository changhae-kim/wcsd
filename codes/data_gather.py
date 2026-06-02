import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt

def get_xas_i0_from_file(filepath, columns, sort_by=None, key_range=(-np.inf, +np.inf)):
    if sort_by is not None:
        key_list = []
    data_list = []
    f = open(filepath, 'rt')
    for line in f:
        if line.startswith('#'):
            continue
        x = line.split()
        data_list.append([float(x[i]) for i in columns])
        if sort_by is not None:
            key_list.append(float(x[sort_by]))
    f.close()
    if sort_by is not None:
        data_list = [d for k, d in sorted(zip(key_list, data_list)) if k >= key_range[0] and k <= key_range[1]]
    return np.array(data_list).swapaxes(0,1).copy()

def get_xas_mu_from_file(filepath, columns, sort_by=None, key_range=(-np.inf, +np.inf)):
    x = get_xas_i0_from_file(filepath, columns, sort_by, key_range)
    energy, i0, it, iff = 0, 1, 2, 3
    n_sign = np.count_nonzero(x[i0] * x[it] <= 0.0)
    if n_sign > 0:
        print(f'{filepath} contains {n_sign} sign-mismatched i0 & it pairs. Replacing ln(i0/it) with zeros...')
    n_zero = np.count_nonzero(x[i0] == 0.0)
    if n_zero > 0:
        print(f'{filepath} contains {n_zero} zero-valued i0. Replacing iff/i0 with zeros...')
    data_list = [
            x[energy],
            np.log(np.divide(x[i0], x[it], out=np.ones_like(x[i0]), where=(x[i0] * x[it] > 0.0), )),
            np.divide(x[iff], x[i0], out=np.zeros_like(x[i0]), where=(x[i0] != 0.0), ),
            ]
    return np.array(data_list)

def get_xas_data(index, columns, batch_size=None, mode='i0', sort_by=None, key_range=(-np.inf, +np.inf)):
    filepaths = []
    batch = []
    f = open(index, 'rt')
    for line in f:
        filepath = line.strip()
        if filepath.startswith('#'):
            continue
        elif filepath == '' and batch != []:
            filepaths.append(batch)
            batch = []
        else:
            batch.append(filepath)
    if batch != []:
        filepaths.append(batch)
    f.close()

    if batch_size is not None:
        batches = []
        for prebatch in filepaths:
            prebatch_size = len(prebatch)
            quotient = prebatch_size // batch_size
            remainder = prebatch_size % batch_size
            for i in range(quotient-1):
                batches.append(prebatch[(i+0)*batch_size:(i+1)*batch_size])
            batches.append(prebatch[(quotient-1)*batch_size:])
        filepaths = batches

    func_dict = {
            'i0': get_xas_i0_from_file,
            'mu': get_xas_mu_from_file,
            }
    get_xas_data_from_file = func_dict[mode]

    print('Loading...')
    data_list = []
    dim_min = 1E99
    dim_max = 0
    for batch in filepaths:
        d_list = []
        d_min = 1E99
        d_max = 0
        for filepath in batch:
            d = get_xas_data_from_file(filepath, columns, sort_by, key_range)
            d_list.append(d)
            if d_min > d.shape[-1]:
                d_min = d.shape[-1]
            if d_max < d.shape[-1]:
                d_max = d.shape[-1]
        data_list.append(d_list)
        if dim_min > d_min:
            dim_min = dim_min
        if dim_max < d_max:
            dim_max = d_max
        if d_min == d_max:
            print(f'Batch starting with {batch[0]}: ({len(d_list)}, {d_list[0].shape[0]}, {d_min})')
        else:
            print(f'Batch starting with {batch[0]}: ({len(d_list)}, {d_list[0].shape[0]}, {d_min}-{d_max})')

    print('Processing...')
    array_list = []
    for i, d_list in enumerate(data_list):
        a_list = []
        for d in d_list:
            pad_left = (dim_max - d.shape[-1]) // 2
            pad_right = dim_max - d.shape[-1] - pad_left
            d_pad = np.pad(d, ((0, 0), (pad_left, pad_right)), mode='reflect')
            a_list.append(d_pad)
        a_list = np.array(a_list).swapaxes(0,1)[None,...].copy()
        array_list.append(a_list)
        print(f'Batch {i}: {a_list.shape}')

    return array_list

def plot_xas_data(index, columns, mode='i0', sort_by=None, key_range=(-np.inf, +np.inf)):
    filepaths = []
    f = open(index, 'rt')
    for line in f:
        filepath = line.strip()
        if filepath != '':
            filepaths.append(filepath)
    f.close()

    func_dict = {
            'i0': get_xas_i0_from_file,
            'mu': get_xas_mu_from_file,
            }
    get_xas_data_from_file = func_dict[mode]

    figpaths = []
    for filepath in filepaths:
        data_array = get_xas_data_from_file(filepath, columns, sort_by, key_range)
        names = filepath.split('/')
        toppath = f'./{names[-3]}'
        dirpath = f'./{names[-3]}/{names[-2]}'
        figpath = f'./{names[-3]}/{names[-2]}/{names[-1][:-4]}.png'
        if not os.path.exists(toppath):
            os.mkdir(toppath)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        plt.figure()
        for x in data_array:
            plt.plot(x)
        plt.tight_layout()
        plt.savefig(figpath)
        figpaths.append(figpath)

    with tarfile.open(f'./{names[-3]}.tgz', 'w:gz') as tar:
        for figpath in figpaths:
            tar.add(figpath)
    for figpath in figpaths:
        os.remove(figpath)

    return

if __name__ == '__main__':
    data_list = get_xas_data('/sdcc/u/ckim6/BNL/Delafossite/db/XAS Cu Ga data/Cu Ga XAS data.idx', [0, 1, 2, 4], mode='mu')
    np.savez('CuGaXAS/data.npz', *data_list)
    data_list = get_xas_data('/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.idx', [-1, 1, 2, 4], mode='mu', sort_by=0)
    np.savez('RuXAS/data.npz', *data_list)
    plot_xas_data('/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.idx', [1, 2, 4], mode='i0', sort_by=0)
    data_list = get_xas_data('/sdcc/u/ckim6/BNL/Delafossite/db/Dali_Yang_20250729.318587.prebin.idx', [0, 1, 2, 4], mode='mu', sort_by=0, key_range=(21997.000000, 22853.403311))
    np.savez('RuXAS/prebin.npz', *data_list)
