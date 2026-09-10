"""Tests for verifying QREs for plaquettes"""

import numpy as np
import pytest
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, PowerOfTwoBatchedHammingWeightPhasing
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.qre import resource_estimator

from psiq_fh.trotterization.fermi_hubbard_data import PlaquetteTermData
from psiq_fh.utils.jw_ordering_utils import generate_even_odd_enum
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, exptXXYY


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
@pytest.mark.parametrize("plaquette_color", ["pink", "gold"])
def test_plaquette_qre(lattice_size, plaquette_color):
    """Verify plaquette rotation count and T-count.

    Note:
        - See Appendix E of 2012.09238
        - Each plaquette will use L^2 rotations (without HWP)
          and 4L^2 T-gates (from cHads). See Ref above.
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites

    # Set up QPU instance:
    qc = QPU(num_qubits=total_num_qubits, filters=[">>witness>>", ">>buffer>>"])

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    # For one step
    evolution_time = 0.83  # sets angle - note if time is pi and coeff is 1 then no rotations in metrics as all simplify
    enumeration = generate_even_odd_enum(lattice_size)

    kinetic_coefficient = 1
    plaq_data = PlaquetteTermData(plaquette_color, enumeration, evolution_time, kinetic_coefficient)

    plaquette_trotter_step = PlaquetteTrotterStep()
    plaquette_trotter_step.compute(psi_reg, plaq_data)

    metrics = qc.metrics()

    assert metrics["rotation_count"] == lattice_size**2

    # Note: each cHad gate will contribute 2 T-gates
    assert qc.witness.filter("qc.had", condition=1).count() == 2 * lattice_size**2


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
@pytest.mark.parametrize("plaquette_color", ["pink", "gold"])
def test_plaquette_qre_with_hwp(lattice_size, plaquette_color):
    """Verify plaquette rotation count and T-count where we
       utilize Hamming weight phasing.

    Note:
        - See Appendix B.2 of 2411.02160 for cost
          of a HWP in catalyzed Trotter.
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 10

    # Set up QPU instance:
    qc = QPU(
        num_qubits=total_num_qubits,
        pre_filters=[
            ">>clean-ladder-filter>>",
            ">>single-control-filter>>",
            ">>witness>>",
        ],
        filters=[">>buffer>>"],
    )

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    # For one step
    evolution_time = 0.91  # sets angle - note if time is pi and coeff is 1 then no rotations in metrics as all simplify
    enumeration = generate_even_odd_enum(lattice_size)

    hw_qbk = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    kinetic_coefficient = 1

    plaq_data = PlaquetteTermData(plaquette_color, enumeration, evolution_time, kinetic_coefficient)

    exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)
    plaquette_trotter_step.compute(psi_reg, plaq_data)

    metrics = resource_estimator(qc).resources(expanded=True)

    # each controlled-Had be decomposed into 2 T-gates
    t_count = metrics["t_gates"]
    assert t_count == 4 * lattice_size**2

    # Analytical expression for Toffolis for generalized phase gradient for 1 Trotter step
    lsquared = lattice_size**2
    w_lsquared = lsquared.bit_count()
    n_toffs_phase_grad = lsquared + np.floor(np.log2(lattice_size**2)).astype(int) - np.array(w_lsquared) + 1
    assert metrics["aggregated_toff_count"] == n_toffs_phase_grad

    # Expect log2(L**2) + 1 for catalyst qubits and 1 for phase grad addition (payload)
    assert metrics["rotations"] == np.floor(np.log2(lattice_size**2)).astype(int) + 1 + 1


def setup_plaquette_circuits_for_hwp_qre_testing(n_batches, lattice_size, plaquette_color):
    """Return circuit metrics from running plaquette circuit
    with or without batched hamming weight phasing.

    Args:
         n_batches (int | None): Number of batches of rotations. If None,
             no HWP used.
         lattice_size (int): Lattice size for Fermi-Hubbard
         plaquette_color (str): Color of plaquette: 'pink' or 'gold'
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 2 + 4

    # Set up QPU instance:
    qc = QPU(
        num_qubits=total_num_qubits,
        pre_filters=[
            ">>clean-ladder-filter>>",
            ">>single-control-filter>>",
            ">>witness>>",
        ],
        filters=[">>buffer>>"],
    )

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    evolution_time = 0.66  # sets angle - note if time is pi and coeff is 1 then no rotations in metrics as all simplify

    if n_batches:
        hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=n_batches)
    else:
        hwp_qbk = None

    kinetic_coefficient = 1

    enumeration = generate_even_odd_enum(lattice_size)
    plaq_data = PlaquetteTermData(plaquette_color, enumeration, evolution_time, kinetic_coefficient)

    exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)
    plaquette_trotter_step.compute(psi_reg, plaq_data)

    if n_batches:
        qc.release_all_rotation_catalyst_qubits()

    metrics = resource_estimator(qc).resources(expanded=True)
    metrics["utilized_qubit_highwater"] = qc.utilized_qubit_highwater
    return metrics


@pytest.mark.parametrize("lattice_size", [8, 10])
@pytest.mark.parametrize("plaquette_color", ["pink", "gold"])
def test_plaquette_term_qres_with_batched_hwp(lattice_size, plaquette_color):
    """Run sanity checks on using one or two batches. Then,
    cross-verify against results in arxiv:2411.02160.
    """
    metrics_no_hwp = setup_plaquette_circuits_for_hwp_qre_testing(
        n_batches=None, lattice_size=lattice_size, plaquette_color=plaquette_color
    )
    metrics_one_batch = setup_plaquette_circuits_for_hwp_qre_testing(
        n_batches=1, lattice_size=lattice_size, plaquette_color=plaquette_color
    )
    metrics_two_batch = setup_plaquette_circuits_for_hwp_qre_testing(
        n_batches=2, lattice_size=lattice_size, plaquette_color=plaquette_color
    )

    # T-gate counts
    assert metrics_no_hwp["t_gates"] == 4 * lattice_size**2
    assert metrics_one_batch["t_gates"] == 4 * lattice_size**2
    assert metrics_two_batch["t_gates"] == 4 * lattice_size**2

    # Not using HWP uses fewest qubits (just synthesizing each rotation)
    assert metrics_no_hwp["utilized_qubit_highwater"] < metrics_one_batch["utilized_qubit_highwater"]

    # Using two batches uses fewer qubits than using one batch (latter uses more ancillae)
    assert metrics_two_batch["utilized_qubit_highwater"] < metrics_one_batch["utilized_qubit_highwater"]
    # But uses more Toffolis (for adders)
    assert metrics_two_batch["aggregated_toff_count"] > metrics_one_batch["aggregated_toff_count"]

    # Cross-verify against paper data (arxiv:2411.02160)
    # under "batched, catalyzed"
    # in batches of 0.5 L^2 rotations (i.e. 2 batches)
    def hw_toff(n_rots_hwp):
        """Number of Toffolis for computing the Hamming weight."""
        return int(n_rots_hwp) - int(n_rots_hwp).bit_count()

    # Toffolis from catalyst addition
    nTof_cat = 2 * (np.floor(np.log2(0.5 * lattice_size**2)) + 1)

    # Toffolis from Hamming weight computation
    nTof_hw = 2 * hw_toff(n_rots_hwp=0.5 * lattice_size**2)

    nTof = nTof_hw + nTof_cat
    assert np.isclose(
        nTof,
        metrics_two_batch["aggregated_toff_count"],
        atol=0.05 * metrics_two_batch["aggregated_toff_count"],
    )
