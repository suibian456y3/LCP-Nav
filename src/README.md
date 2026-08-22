# LCP-Nav training code

This directory contains the compact training implementation used by the LCP-Nav study.

## Main components

- `vint_train/models/dvn/dvn.py`: deterministic candidate trajectories, latent rollout and endpoint scoring
- `vint_train/models/regularizers.py`: SIGReg non-collapse regularization
- `vint_train/training/`: training, evaluation and loss utilities
- `config/`: experiment configurations

The code expects the standard goal-conditioned navigation data layout used by the ViNT/GNM training stack. Dataset paths in the YAML files are machine-specific and must be changed for a new workstation.

## Installation

```bash
conda env create -f train_environment.yml
conda activate nomad_train
pip install -e .
```

## Training entry point

```bash
python train.py -c config/dvn_sigreg_go_stanford_15.yaml
```

The command is a protocol template. It requires the corresponding dataset, GPU configuration and checkpoint/output directories to be configured locally first.
