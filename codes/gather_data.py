"""
Gather XAS spectra and package them into NumPy arrays for training.

Input:
- An index file containing one data file path per line.
- File paths are relative to the index file.
- Blank lines separate batches.
- Lines beginning with '#' are ignored.

Processing:
- Selected columns are read from each ASCII data file.
- For mu mode, the selected columns should be:
  [energy, i0, it, iff]
  The derived channels are:
  [energy, log(i0/it), iff/i0]
- Spectra are truncated to a common length.

Output:
- A list of arrays shaped [1, channels, timesteps, energies],
  suitable for saving with np.savez(...).

Example:

    python gather_data.py ../data/index
"""

import argparse
import os
import numpy as np

def get_xas_i0_from_file(filepath, columns):
    """Read selected columns from an ASCII file."""
    data_list = []
    with open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            x = line.split()
            data_list.append([float(x[i]) for i in columns])
    return np.array(data_list).swapaxes(0,1).copy()

def get_xas_mu_from_file(filepath, columns):
    """Build [energy, log(i0/it), iff/i0] channels from an ASCII file."""
    x = get_xas_i0_from_file(filepath, columns)
    energy, i0, it, iff = 0, 1, 2, 3
    data_list = [x[energy], np.log(x[i0]/x[it]), x[iff]/x[i0]]
    return np.array(data_list)

def get_xas_data(index, columns, mode="i0"):
    """Read an index file and return a list of arrays."""
    topdir = os.path.dirname(index)
    filepaths = []
    batch = []
    with open(index, "rt") as f:
        for line in f:
            filepath = line.strip()
            if filepath.startswith("#"):
                continue
            elif filepath == "" and batch != []:
                filepaths.append(batch)
                batch = []
            else:
                batch.append(os.path.join(topdir, filepath))
        if batch != []:
            filepaths.append(batch)

    func_dict = {
            "i0": get_xas_i0_from_file,
            "mu": get_xas_mu_from_file,
            }
    get_xas_data_from_file = func_dict[mode]

    print("Loading...")
    data_list = []
    dim_min = float("inf")
    for batch in filepaths:
        d_list = []
        d_min = float("inf")
        d_max = 0
        for filepath in batch:
            d = get_xas_data_from_file(filepath, columns)
            d_list.append(d)
            if d_min > d.shape[-1]:
                d_min = d.shape[-1]
            if d_max < d.shape[-1]:
                d_max = d.shape[-1]
        data_list.append(d_list)
        if d_min == d_max:
            print(f"Batch starting with {batch[0]}: ({len(d_list)}, {d_list[0].shape[0]}, {d_min})")
        else:
            print(f"Batch starting with {batch[0]}: ({len(d_list)}, {d_list[0].shape[0]}, {d_min}-{d_max})")
        if dim_min > d_min:
            dim_min = d_min

    print("Processing...")
    array_list = []
    for i, d_list in enumerate(data_list):
        a_list = [d[..., :dim_min] for d in d_list]
        a_list = np.array(a_list).swapaxes(0, 1)[None, ...].copy()
        array_list.append(a_list)
        print(f"Batch {i}: {a_list.shape}")

    return array_list

def main():
    parser = argparse.ArgumentParser(
        description="Gather XAS data files and save the batches as an NPZ archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument("index", type=str, help="Path to the index file")
    parser.add_argument("-o", "--output", type=str, default="data.npz",
        help="Path to the output NPZ file (default=data.npz)")
    parser.add_argument("-c", "--columns", type=int, nargs="+", default=[0, 1, 2, 4],
        help="Zero-based columns to read from each data file (default=[0, 1, 2, 4])")
    parser.add_argument("-m", "--mode", type=str, choices=("i0", "mu"), default="mu",
        help="Channel construction mode (default=mu)")
    args = parser.parse_args()

    data_list = get_xas_data(args.index, args.columns, mode=args.mode)
    np.savez(args.output, *data_list)

if __name__ == "__main__":
    exit(main())
