# learning-nmpc-q-loss
Imitation Learning from NMPC via acados and PyTorch

This repo will contain the implementation described in the paper:
`Imitation Learning from Nonlinear MPC via the Exact Q-Loss and its Gauss-Newton Approximation`, available [here](https://cdn.syscop.de/publications/Ghezzi2023b.pdf).

The paper will be presented in December 2023 at the IEEE Conference on Decision and Control. The code will be published within that date.

## Installation:

Running `python 3.8.12`

### 1. Installation

`pip install -e .`

`pre-commit install`

### 2. Acados installation

Follow the instruction at: [acados docs](docs.acados.org) \
After installing acados on your machine, install via pip the acados python interface in the python environment you use to run this repo as follow \
`pip install -e <acados_root>/interfaces/acados_template`
