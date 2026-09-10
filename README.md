# PsiQuantum Fermi Hubbard: Qubricks for Trotterized Fermi Hubbard in active volume architectures

This repository accompanies the paper "Compiling the 2D Fermi-Hubbard ground-state energy estimation algorithm for active volume architectures" ([arxiv:2609.05316](https://arxiv.org/abs/2609.05316)).

Unlike a number of previous resource estimates that neglect the controlled implementation of the time-evolution circuits, our estimates explicitly include the control structure required for QPE. To improve the reproducibility of our resource estimates all circuits are implemented in the open-access [PsiQuantum Development Kit](https://construct.psiquantum.com/qdk). PsiQDK compiles the circuits, reports gate counts, and computes active volume using an implementation of the lookup tables of [Litinski et al.](https://arxiv.org/abs/2211.15465), extended as described in Section 3 of the accompanying paper.

<img src="./examples/trotter/figures/big_summary_fig.jpeg" width="700" alt="Summary figure">

To compile these circuits and obtain the corresponding resources estimates, custom [qubricks](https://construct.psiquantum.com/docs/psiqdk-workbench/new-tutorials/Qubricks.html) that implement the Trotterized time evolution under the Fermi-Hubbard model Hamiltonian were built and tested. We provide examples on how to generate resource estimation data for prior state-of-the-art as well as the circuit proposed in the paper.

We hope that this code is useful beyond our examples. 

We note the following limitations. The qubricks are limited to even rectangular lattice sizes, and Hamming weight phasing can only be batched in powers of two. If you would like to run circuits for lattice dimensions, batch sizes, or compilation choices that differ from the paper, you will have to supply the corresponding error budgeting data.

Code to produce valid error budgeting data in accordance with the paper is not currently open-sourced.
Code to schedule logical blocks to obtain logical cycle resource estimates and runtimes is not currently open-sourced.
We hope that these additions will come in time!



## Install

For installation, first clone this repository onto your local machine. We have used `uv` ([link](https://docs.astral.sh/uv/)) as a package manager and we highly recommend installing `uv` locally to use this package.


For a standard installation, pulling in the latest requirements for your version of Python
```sh
uv venv && uv pip install -e .

# python / pip equivalent
# python -m venv .venv && source .venv/bin/activate
# pip install -e .
```

To reproduce the paper's numbers, use the exact environment they were produced in rather
than the loose bounds in `pyproject.toml`. This is natively done in `uv` via the `uv.lock` file, but we have also provided a `reproduction_requirements.txt` file.

```sh
uv sync --frozen

# python / pip equivalent
# python -m venv .venv && source .venv/bin/activate
# pip install -r reproduction_requirements.txt && pip install -e . --no-deps
```

Note that we only guarantee `uv sync --frozen` as a way to reproduce the numbers in the paper: the pinned version of `psiqdk` fixes the active-volume lookup table used. As `psiqdk` is in active development future versions may produce different values.

## Reproducing the results

We have provided notebooks as tutorials on our work and to re-generate the data and plots used in the paper. We recommend running the three notebooks in `examples/trotter/` in the following order: 
1. `generate_baseline_data.ipynb`
2. `generate_improved_data.ipynb`, 
3. `reproduce_plots.ipynb`. 

The JSONs in `data/trotter/error_budgeting_data/` are the only inputs.


`generate_baseline_data.ipynb` and `generate_improved_data.ipynb` read the input JSONs and write resource counts into `examples/trotter/paper_data_and_plots/`. `reproduce_plots.ipynb` reads those resource counts and produces the figures.

All of the requisite input and outputs come populated, so you are able to run `reproduce_plots.ipynb` once the package is installed. The `generate_*` notebooks sweep lattice sizes 4-20 and take 2-3 minutes to complete.

## Tests

The test suite can be ran via:
```bash
uv run pytest
```

## Citation

If you have used the results provided here in your work, please cite the paper:
```bibtex
@misc{apel2026compiling2dfermihubbardgroundstate,
  title={Compiling the 2D Fermi-Hubbard ground-state energy estimation
         algorithm for active volume quantum architectures},
  author={Harriet Apel and Athena Caesura and Carys Harvey and Sam Heavey and
          Angus Kan and Jessica Lemieux and Ryan Levy and Sam Pallister and
          Joseph Peetz and William Pol and Sukin Sim and William A. Simon and
          Mark Steudtner and Gideon Uchehara},
  year={2026},
  eprint={2609.05316},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2609.05316},
}
```

If you have used some of this software in your work, please cite this repository as follows:
```bibtex
@software{PsiQ_FH,
  author = {Apel, Harriet and Sim, Sukin},
  title = {{PsiQuantum Fermi Hubbard: Qubricks for trotterized Fermi Hubbard
            in active volume architectures}},
  url = {TODO},
  year = {2026}
}
```
