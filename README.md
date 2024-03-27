# learning-nmpc-q-loss
Imitation Learning from NMPC via acados and PyTorch

This repo will contain the implementation described in the paper:
`Imitation Learning from Nonlinear MPC via the Exact Q-Loss and its Gauss-Newton Approximation`, available [here](https://publications.syscop.de/Ghezzi2023b.pdf).

The paper was presented in December 2023 at the IEEE Conference on Decision and Control.

## Getting started:

Running `python 3.8.12`

### 1. Installation

`pip install -e .`

`pre-commit install`

### 2. Acados installation

Follow the instruction at: [acados docs](docs.acados.org) \
After installing acados on your machine, install via pip the acados python interface in the python environment you use to run this repo as follow \
`pip install -e <acados_root>/interfaces/acados_template`

## How to use:
1. Folder `imitate` contains the routines for creating the model, formulating/solving the MPC problem and running imitation learning.
2. Folder `script` contains different code snippets for training and evaluating the policies, and plotting the results.
3. Folder `tests` contains two python scripts to test the correct behavior of the provided code, especially the dependancy on `acados`.

#### Acados speedup for Linux machines
For Linux machines it is possible to compile the `acados` python interface using `cython`. To do so set the flag `no-cython=False` in the `ArgumentParser` in `main_training.py`.
