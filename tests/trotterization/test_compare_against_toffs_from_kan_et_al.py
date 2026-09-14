"""Cross-verification with QRE expressions from arxiv:2411.02160."""

import numpy as np
from psiq_fh.wb_program.utils import compute_T_rotation_synthesis_mixed_fallback

from psiq_fh.wb_program import FermiHubbardQreRunner
from psiq_fh.wb_program.utils import extract_all
import pytest


def compute_toff_counts_from_kan_et_al(lattice_sizes, queries, trotter_steps, eps_cats, eps_noncats):
    """Compute Toffoli counts from Kan et al (arxiv:2411.02160) expressions. Test for small lattice sizes.

    Args:
        lattice_sizes: Lattice sizes.
        queries: Number of queries.
        trotter_steps: Number of Trotter steps.
        eps_cats: Error per catalyst rotation.
        eps_noncats: Error per non-catalyst rotation.
    """
    # Toffolis from catalyzed HWPs. See Eq (71) in arxiv:2411.02160v2.
    lattice_sizes = np.asarray(lattice_sizes)
    w_sq = np.array([int(L**2).bit_count() for L in lattice_sizes])
    n_toff = (4 * trotter_steps + 1) * (lattice_sizes**2 + (np.floor(np.log2(lattice_sizes**2))).astype(int) - w_sq + 1)

    # Catalyst RZ rotations. See Eq (73) in arxiv:2411.02160v2.
    # Note: corrected number of catalysts in IPG
    n_cats = 2 * (np.floor(np.log2(lattice_sizes**2))).astype(int) + 4
    n_T1 = n_cats * compute_T_rotation_synthesis_mixed_fallback(eps_cats)

    # Non-catalyst RZ rotations. See Eq (74) in arxiv:2411.02160v2.
    n_noncats = 4 * trotter_steps + 1
    n_T2 = n_noncats * compute_T_rotation_synthesis_mixed_fallback(eps_noncats)

    # Total T counts. See Eq (75) in arxiv:2411.02160v2.
    n_T_from_chads = 12 * trotter_steps * lattice_sizes**2  # in two-mode FFFTs
    n_T = n_T_from_chads + n_T2  # these will be multiplied by the number of queries

    # Total Toff cost (from T's and rotations as well)
    toff_total_kan_et_al = n_toff * queries
    toff_total_kan_et_al += (n_T * queries) / 2  # 1 CCZ state -> 2 T states
    toff_total_kan_et_al += n_T1 / 2  # add catalyst cost
    return toff_total_kan_et_al


@pytest.mark.parametrize("lattice_sizes", [[4], [6]])
def test_compare_against_kan_et_al(lattice_sizes):
    """Cross-verify baseline circuit Workbench (WB) Toffoli counts against Kan et al (arxiv:2411.02160) expressions.

    Args:
        lattice_sizes: Lattice sizes.
    """
    # Instantiate baseline WB circuit and retrieve metrics data
    runner = FermiHubbardQreRunner(lattice_sizes=lattice_sizes, variant="baseline", num_batches=1)
    metrics_data = runner.retrieve()

    t_gates_wb = extract_all(metrics_data, "t_gates")
    toffs_wb = extract_all(metrics_data, "aggregated_toff_count")

    T_TO_TOFF = 2  # for comparing against Kan et al. Assumes all T states catalyzed by CCZ states.
    toffs_from_t_gates_wb = t_gates_wb / T_TO_TOFF
    total_toff_counts_wb = toffs_wb + toffs_from_t_gates_wb

    # Compute error fractions from WB circuit data for Kan et al expressions
    eps_cats = np.zeros(len(lattice_sizes))
    eps_noncats = np.zeros(len(lattice_sizes))
    trotter_steps = np.zeros(len(lattice_sizes))
    queries = np.zeros(len(lattice_sizes))

    for i, x_dim in enumerate(lattice_sizes):
        circuit_data = runner.load_circuit_data(x_dim, x_dim, raw=True)
        eps_cats[i] = 2**-circuit_data.cat_bop
        eps_noncats[i] = 2**-circuit_data.rot_bop
        trotter_steps[i] = circuit_data.no_trotter_steps
        queries[i] = circuit_data.no_queries

    toff_total_kan_et_al = compute_toff_counts_from_kan_et_al(
        lattice_sizes, queries, trotter_steps, eps_cats, eps_noncats
    )

    assert np.allclose(total_toff_counts_wb, toff_total_kan_et_al)
