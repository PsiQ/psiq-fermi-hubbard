# Fermi-Hubbard Trotterization qubricks

This repo accompanies the paper "Compiling the 2D Fermi-Hubbard ground-state energy estimation algorithm for active volume architectures" (arxiv:XXXX.XXXXX).

Unlike a number of previous resource estimates that neglect the controlled implementation of the time-evolution circuits, our estimates explicitly include the control structure required for QPE. To improve the reproducibility of our resource estimates, all circuits are implemented in the open-access [PsiQuantum software package](https://construct.psiquantum.com/qdk), which compiles the circuits, reports gate counts, and computes active volume using an implementation of the lookup tables of [Litinski et al.](https://arxiv.org/abs/2211.15465), extended as described in Section 3 of the accompanying paper.

<img src="./examples/trotter/figures/big_summary_fig.jpeg" width="700" alt="Summary figure">

To compile these circuits and obtain the corresponding resources estimates, custom [qubricks](https://construct.psiquantum.com/docs/psiqdk-workbench/new-tutorials/Qubricks.html) that implement the Trotterized time evolution under the Fermi-Hubbard model Hamiltonian were built and tested. We provide examples on how to generate resource estimation data for prior state-of-the-art as well as the circuit proposed in the paper.

We hope that this code is useful beyond our examples. We note the following limitations. The qubricks are limited to even rectangular lattice sizes, and Hamming weight phasing can only be batched in powers of two. If you would like to run circuits for lattice dimensions, batch sizes, or compilation choices that differ from the paper, you will have to supply the corresponding error budgeting data.

Code to produce valid error budgeting data in accordance with the paper is not currently open-sourced.
Code to schedule logical blocks to obtain logical cycle resource estimates and runtimes is not currently open-sourced.
We hope that these additions will come in time!



## Install

Requires Python 3.11–3.13. Everything resolves from public PyPI.

```sh
pip install -e .
```

To reproduce the paper's numbers, use the exact environment they were produced in rather
than the loose bounds in `pyproject.toml`:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r reproduction_requirements.txt
pip install -e . --no-deps
```

The pinned `psiqdk` version matters: it fixes the active-volume lookup table used by the paper.

## Reproducing the results

To read through the mini tutorials and re-generate the data and plots, we recommend running the three notebooks in the following order: `generate_baseline_data.ipynb`, `generate_improved_data.ipynb`, then `reproduce_plots.ipynb`. The 31 JSONs under `error_budgeting_data/` are the only inputs:

1. `generate_baseline_data.ipynb` and `generate_improved_data.ipynb` read the input JSONs and write
   resource counts into `paper_data_and_plots/`.
2. `reproduce_plots.ipynb` reads `paper_data_and_plots/` and produces the figures.

`paper_data_and_plots/` ships populated, so you can run step 2 on its own. Step 1 sweeps lattice
sizes 4 to 20 and takes 2-3 minutes.

## Tests

```sh
pip install -e . --group dev
pytest tests -n auto
```

## Citation

We encourage use of this repo with the appropriate citation:

BIBTEX TODO
