# Parameter-Efficient Continual Learning for Automatic Speech Recognition

This repository contains supplementary material for the paper:

**Parameter-Efficient Continual Learning for Automatic Speech Recognition**, accepted at **Interspeech 2026**.

The repository contains code and configuration files for parameter-efficient continual learning (PECL) experiments in ASR, including LoRA-based baselines, SSVD, and Continual SSVD (CSSVD).

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{vandereeckt2026pecl_asr,
  title     = {Parameter-Efficient Continual Learning for Automatic Speech Recognition},
  author    = {Vander Eeckt, Steven and Van hamme, Hugo},
  booktitle = {Proceedings of Interspeech 2026},
  year      = {2026},
  address   = {Sydney, Australia},
}
```

## Overview

This work studies parameter-efficient continual learning for ASR. The goal is to sequentially adapt a pretrained ASR model to new tasks while training only a small number of parameters and reducing catastrophic forgetting on previously learned tasks.

The proposed method, **Continual SSVD (CSSVD)**, builds on structured Singular Value Decomposition (SSVD). For each adapted linear layer, the pretrained weight matrix is decomposed into singular subspaces. CSSVD protects the dominant singular directions and restricts adaptation to rotations within the low-energy tail subspace. For later tasks, adapted weights are merged through averaging to further reduce forgetting.

## Repository structure

```text
.
├── code/
├── conf/
├── models/
└── README.md
```

## Code

The models are trained using **ESPnet2**.

The code in this repository is intended to be added to ESPnet. In particular, the layer implementations should be placed under:

```text
espnet2/layers/
```

Each baseline is implemented as a separate class that inherits from:

```python
torch.nn.Linear
```

The implementations include LoRA-based methods, SSVD, and CSSVD. SSVD and CSSVD are both implemented within `ssvdolib`; the option `knowledge_preserve_mode` determines which variant is used.

Adapter insertion is handled by:

```text
create_adapter.py
create_adapter_fn.py
```

These files already exist in ESPnet, but we extended them to support additional LoRA and SSVD-based adapters.

## Configuration files

The `conf/` directory contains the configuration files used for training the different methods.

For all methods, adaptation is restricted to the weight matrices of linear layers. All other parameters are frozen. This includes the rest of the ASR model and the output layers.

