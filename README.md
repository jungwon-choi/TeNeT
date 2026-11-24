<div align="center">

# TeNeT: Temporal Neural Transformer

**Learning Dynamic Brain Connectome with Graph Transformers for Psychiatric Diagnosis Classification**

Byung-Hoon Kim<sup>\*</sup>, Jungwon Choi<sup>\*</sup>, EungGu Yun, Kyungsang Kim, Xiang Li, Juho Lee

<sup>\*</sup>Equal contribution

[![ISBI 2024](https://img.shields.io/badge/ISBI-2024%20Oral-1f6feb)](https://ieeexplore.ieee.org/document/10635508)
[![NeurIPS 2023 TGL](https://img.shields.io/badge/NeurIPS%202023-TGL%20Workshop-8957e5)](https://openreview.net/forum?id=WvFHDkIhmp)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-2ea44f)](LICENSE)

</div>

<p align="center">
    <img src="assets/connectome_embedding.png" width="90%">
</p>

TeNeT is a graph Transformer that learns the representation of **dynamic functional connectivity (FC)** from resting-state fMRI. It embeds the position, structure, and time of the brain functional connectivity graph at each sliding window, and learns their relationships with Transformers across space and time. TeNeT is trained and validated on over 50,000 FC samples from ten public datasets for phenotype prediction and psychiatric diagnosis classification.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Overview

<p align="center">
    <img src="assets/scheme.png" width="55%">
</p>

**Connectome embedding.** For each sliding window *t*, TeNeT builds a connectome embedding *X<sub>t</sub>* ∈ ℝ<sup>(N+1)×D</sup> from
- **structure**: the windowed correlation matrix *R<sub>t</sub>* without self-loops,
- **position**: the one-hot encoding of each ROI, and
- **time**: a GRU encoding of the ROI time series up to the last time point of the window.

**Spatial and temporal Transformers.** TeNeT is a composition *f = h ∘ g* of two Transformer encoders.
- *g* attends across ROIs within each window. At every layer, the FC matrix and node degree are linearly projected and concatenated to the self-attended features before the MLP, so the graph structure is injected regardless of depth.
- *h* attends across windows and produces the subject representation *h*<sub>dyn</sub>, followed by a linear head for classification or regression.

## Installation

```bash
git clone https://github.com/jungwon-choi/TeNeT.git
cd TeNeT

conda create -n tenet python=3.10 -y
conda activate tenet

pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The code is tested with Python 3.10 and PyTorch 2.2 (CUDA 12.1), and also runs on CPU.

## Data preparation

Place the ROI time series and the phenotype table of each dataset under `data/`:

```
data/
└── hcpep-rest/
    ├── hcpep-rest_roi-schaefer400.pkl   # dict {subject_id: ndarray [T, N]}
    └── hcpep-rest.csv                   # 'subject_id' column + target columns
```

| File | Format |
|---|---|
| `{dataset}_roi-{roi}.pkl` | Dictionary from subject ID (str) to an array of shape `[time points, ROIs]`, the mean BOLD signal within each ROI of the Schaefer atlas (400 ROIs in the paper). Each ROI is z-scored per subject when loaded. |
| `{dataset}.csv` | One row per subject with a `subject_id` column and the target column named in the config (e.g., `sex`, `interview_age`, `phenotype`). |

The paper uses the following datasets. Preprocessing followed the pipeline provided by each dataset, with images registered to the MNI space; for subjects with multiple sessions, only the first session is used.

| Dataset | Name | Subjects | Diagnosis (P / C) | Tasks |
|---|---|---:|---:|---|
| [UK Biobank](https://www.ukbiobank.ac.uk/) | `ukb-rest` | 40,913 | – | gender, age |
| [ABCD](https://abcdstudy.org/) | `abcd-rest` | 9,111 | – | gender |
| [HCP-YA](https://www.humanconnectome.org/study/hcp-young-adult) | `hcp-rest` | 1,093 | – | gender |
| [HCP-D](https://www.humanconnectome.org/study/hcp-lifespan-development) | `hcpd-rest` | 632 | – | gender, age |
| [HCP-A](https://www.humanconnectome.org/study/hcp-lifespan-aging) | `hcpa-rest` | 723 | – | gender, age |
| [HCP-EP](https://www.humanconnectome.org/study/human-connectome-project-for-early-psychosis) | `hcpep-rest` | 176 | 120 / 56 | gender, age, dx |
| [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide) | `abide-rest` | 884 | 408 / 476 | gender, age, dx |
| [ADHD200](https://fcon_1000.projects.nitrc.org/indi/adhd200) | `adhd200-rest` | 669 | 280 / 389 | gender, age, dx |
| [UCLA](https://openneuro.org/datasets/ds000030) | `ucla-rest` | 265 | 140 / 125 | gender, age, dx<sup>†</sup> |
| [COBRE](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html) | `cobre-rest` | 173 | 89 / 84 | gender, age, dx<sup>†</sup> |

<sup>†</sup> Multi-class diagnosis. Access to each dataset follows its own data use agreement. For UK Biobank, downstream evaluation uses a fixed 10% subset of subjects.

## Training and evaluation

Each config in `configs/` is named `<dataset>_<task>.yaml` (`gender`, `age`, or `dx`) and inherits the shared settings in `configs/default.yaml`, which follow the paper.

**5-fold cross-validation on a single GPU**

```bash
python train.py --config configs/hcpep_dx.yaml --data_dir ./data --output_dir ./result/hcpep_dx
```

**Selected folds only**

```bash
python train.py --config configs/hcpep_dx.yaml --data_dir ./data --output_dir ./result/hcpep_dx --folds 0 1
```

**Multiple GPUs** (`minibatch_size` is split across GPUs)

```bash
torchrun --nproc_per_node 2 train.py \
    --config configs/hcpep_dx.yaml --data_dir ./data --output_dir ./result/hcpep_dx
```

Any config value can be overridden from the command line, e.g. `--lr 1e-5 --minibatch_size 8`.

**Outputs** in `--output_dir`:

| File | Content |
|---|---|
| `results.csv` | Test metrics of every fold with mean and standard deviation |
| `fold<k>/test_metrics.json` | Test metrics at the last epoch |
| `fold<k>/test_predictions.npz` | Subject IDs, labels, predictions, and probabilities on the test fold |
| `fold<k>/train_log.jsonl` | Training and test metrics per evaluated epoch |
| `fold<k>/model.pth` | Model weights |
| `train.log`, `config.json` | Full log and resolved config |

Classification reports AUROC (macro one-vs-rest for multi-class), accuracy, balanced accuracy, precision, recall, and F1. Regression reports R², MSE, and MAE.

## Repository structure

```
TeNeT/
├── train.py              # k-fold training and evaluation (single GPU / torchrun)
├── tenet/
│   ├── model.py          # TeNeT: connectome embedding, graph and temporal Transformers
│   ├── data.py           # fMRI dataset and cross-validation splits
│   ├── metrics.py        # classification and regression metrics
│   └── config.py         # YAML config loading and command-line overrides
├── configs/
│   ├── default.yaml      # shared settings
│   └── <dataset>_<task>.yaml
├── assets/
└── requirements.txt
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2024learning,
  title        = {Learning dynamic brain connectome with graph transformers for psychiatric diagnosis classification},
  author       = {Kim, Byung-Hoon and Choi, Jungwon and Yun, EungGu and Kim, Kyungsang and Li, Xiang and Lee, Juho},
  booktitle    = {2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages        = {1--5},
  year         = {2024},
  organization = {IEEE}
}

@inproceedings{kim2023large,
  title     = {Large-scale Graph Representation Learning of Dynamic Brain Connectome with Transformers},
  author    = {Kim, Byung-Hoon and Choi, Jungwon and Yun, EungGu and Kim, Kyungsang and Li, Xiang and Lee, Juho},
  booktitle = {Temporal Graph Learning Workshop (NeurIPS 2023)},
  year      = {2023}
}
```

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
