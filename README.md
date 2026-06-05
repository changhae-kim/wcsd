# Weight-Centric Spectrum Denoiser (WCSD)

WCSD is a lightweight self-supervised neural network for denoising X-ray absorption spectroscopy (XAS) data. The model employs blind-spot convolutions and an ensemble of convolutional subnetworks with learnable mixing weights to suppress noise while preserving spectral features.

This repository provides:

* tools for assembling XAS datasets,
* training utilities for WCSD,
* denoising workflows for XAS spectra,
* XAFS normalization using Larch,
* a Cu/Ga XAS benchmark dataset,
* pretrained model weights used in the accompanying publication.

---

## Citation and Related Work

WCSD adapts the Weight-Centric Image/Video Denoiser (WCID/WCVD) architecture to spectral sequences.

If you use this software or dataset in published work, please cite the accompanying publication. Users of the weight-centric denoising methods should also cite the original WCID/WCVD publication.

```bibtex
Citation information for the WCSD work will be added upon publication.

@misc{lee2025machinelearningpipelinedenoising,
      title={Machine Learning Pipeline for Denoising Low Signal-To-Noise Ratio and Out-of-Distribution Transmission Electron Microscopy Datasets},
      author={Brian Lee and Meng Li and Judith C Yang and Dmitri N Zakharov and Xiaohui Qu},
      year={2025},
      eprint={2512.04045},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2512.04045},
}
```

---

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate wcsd
```

Install the package:

```bash
pip install -e .
```

This provides the following command-line tools:

```text
gather_data
train_net
norm_xas
```

---

## Repository Structure

The repository includes a Cu/Ga XAS dataset together with a pretrained WCSD model for reproducing the examples reported in the accompanying publication.

```text
.
├── examples/
│   └── Cu Ga XAS data/
│       ├── best.pt
│       ├── config.json
│       ├── index
│       └── */*.dat
│
├── wcsd/
│   ├── model/
│   │   └── net.py
│   ├── scripts/
│   │   ├── gather_data.py
│   │   ├── train_net.py
│   │   └── norm_xas.py
│   └── utils/
│       └── config.py
├── environment.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Data Files

The directory

```text
examples/Cu Ga XAS data/
```

contains:

| File          | Description                                   |
| ------------- | --------------------------------------------- |
| `best.pt`     | Pretrained WCSD model weights                 |
| `config.json` | XAS normalization configuration               |
| `index`       | Index file listing the spectra in the dataset |
| `*/*.dat`     | Raw XAS spectra                               |

---

## Workflow

The typical workflow consists of these stages:

```text
Raw spectra
    │
    ▼
gather_data
    │
    ▼
Training archive (.npz)
    │
    ▼
train_net
    │
    ▼
WCSD model (.pt)
    │
    ▼
norm_xas
    │
    ▼
Denoised and normalized spectra
```

---

## Reproducing the Published Example

To denoise and normalize the supplied spectra using the pretrained model weights:

```bash
norm_xas \
    -f "examples/Cu Ga XAS data/index" \
    -c "examples/Cu Ga XAS data/config.json" \
    -t muff \
    -m "examples/Cu Ga XAS data/best.pt" \
    -i 2
```

The options `-t muff` and `-i 2` select the fluorescence channel (`iff/i0`) used in the published model.

This command:

1. loads the spectra listed in `index`,
2. denoises the selected channels using the pretrained WCSD model,
3. performs XAFS normalization using Larch,
4. writes normalized spectra to `flat.tgz`.

The output archive contains one normalized spectrum file (`flat_*.dat`) for each input spectrum.

---

## Input Data Format

### Dataset Generation

For `mode=mu`, the selected columns are expected to contain:

```text
energy  i0  it  iff
```

These channels are transformed into:

```text
energy
ln(i0 / it)
iff / i0
```

before being stored in the training archive.

### Normalization Configuration

The normalization workflows use a JSON configuration file.

Example:

```json
{
    "labels": "energy i0 it ir iff aux1 aux2 aux3 aux4",
    "pre_edge_kws": {
        "pre2": -150.0,
        "pre1": -50.0,
        "nnorm": 4
    }
}
```

---

## Generating a Training Dataset

Training data are generated from an index file containing one data file path per line.

Blank lines separate batches. Spectra within the same batch are treated as repeated measurements of the same sample and are processed together during training.

Example:

```text
sample_001.dat
sample_002.dat
sample_003.dat

sample_101.dat
sample_102.dat
```

Generate a training archive:

```bash
gather_data /path/to/index data.npz
```

The output archive contains one NumPy array per batch and can be used directly for training.

### Included Dataset

The published Cu/Ga dataset can be converted into a training archive with:

```bash
gather_data "examples/Cu Ga XAS data/index" data.npz
```

---

## Training WCSD

Train a model from a prepared dataset:

```bash
train_net -f data.npz -i 2
```

Model checkpoints are written as:

```text
best.pt
final.pt
```

where:

* `best.pt` corresponds to the lowest reconstruction loss,
* `final.pt` corresponds to the final training epoch.

---

## Acknowledgement

This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

---

## Disclaimer

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
