"""Tests for verifying QREs for interaction part"""

import numpy as np
import pytest
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, PowerOfTwoBatchedHammingWeightPhasing
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.qre import resource_estimator

from psiq_fh.trotterization.fermi_hubbard_data import InteractionTermData
from psiq_fh.utils.jw_ordering_utils import generate_high_low_enum
from psiq_fh.trotterization.interaction import InteractionTrotterStep
from psiq_fh.utils.fermi_hubbard_hamiltonian import get_expected_norm


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
@pytest.mark.parametrize("particle_hole_symmetry", [True, False])
def test_interaction_qre(lattice_size, particle_hole_symmetry):
    """Verify QRE for interaction term for a single Trotter step.

    Note:
        - See Appendix E of 2012.09238
        - Each interaction will use L^2 rotations (without HWP).
          See Ref above.
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites

    # Set up QPU instance:
    qc = QPU(filters=[">>witness>>", ">>buffer>>"])
    qc.reset(total_num_qubits)

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    evolution_time = np.pi
    potential_coefficient = 8

    norm = get_expected_norm(
        lattice_size, lattice_size, potential_coefficient, particle_hole_symmetry=particle_hole_symmetry
    )

    enumeration = list(generate_high_low_enum(lattice_size))
    normalized_term_coefficient = potential_coefficient / (4 * norm)
    interaction_data = InteractionTermData(
        enumeration, evolution_time, normalized_term_coefficient, particle_hole_symmetry
    )

    interaction_trotter_step = InteractionTrotterStep()

    interaction_trotter_step.compute(psi_reg, interaction_data)

    metrics = resource_estimator(qc).resources(expanded=True)

    if particle_hole_symmetry:
        assert metrics["rotations"] == lattice_size**2
    else:
        assert metrics["rotations"] == 3 * lattice_size**2


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
def test_interaction_qre_with_hwp(lattice_size):
    """Verify QRE for interaction term for a single Trotter step
       using Hamming weight phasing.

    Note:
        - See Appendix B.2 of 2411.02160 for cost
          of a HWP in catalyzed Trotter
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 5

    # Set up QPU instance:
    qc = QPU(
        pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"], filters=[">>buffer>>"]
    )
    qc.reset(total_num_qubits)

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    evolution_time = 0.34
    potential_coefficient = 8

    norm = get_expected_norm(lattice_size, lattice_size, potential_coefficient, particle_hole_symmetry=True)

    hw_qbk = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    enumeration = list(generate_high_low_enum(lattice_size))
    normalized_term_coefficient = potential_coefficient / (4 * norm)
    interaction_data = InteractionTermData(enumeration, evolution_time, normalized_term_coefficient)
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    interaction_trotter_step.compute(psi_reg, interaction_data)
    metrics = resource_estimator(qc).resources(expanded=True)

    assert metrics["t_gates"] == 0

    # Analytical expression for Toffs for generalized phase gradient for 1 Trotter step
    lsquared = lattice_size**2
    w_lsquared = lsquared.bit_count()
    n_toffs_phase_grad = lsquared + np.floor(np.log2(lattice_size**2)).astype(int) - np.array(w_lsquared) + 1
    assert metrics["aggregated_toff_count"] == n_toffs_phase_grad

    # Expect log2(L**2) + 1 for catalyst qubits and 1 for phase grad addition (payload)
    assert metrics["rotations"] == np.floor(np.log2(lattice_size**2)).astype(int) + 1 + 1


def setup_interaction_circuits_for_hwp_qre_testing(n_batches, lattice_size):
    """Return circuit metrics from running interaction circuit
    with or without batched hamming weight phasing.

    Args:
         n_batches (int | None): Number of batches of rotations. If None,
             no HWP used.
         lattice_size (int): Lattice size for Fermi-Hubbard
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 2 + 4

    # Set up QPU instance:
    qc = QPU(
        pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"], filters=[">>buffer>>"]
    )
    qc.reset(total_num_qubits)

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    evolution_time = 0.43
    potential_coefficient = 8

    norm = get_expected_norm(lattice_size, lattice_size, potential_coefficient)

    if n_batches:
        hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=n_batches)
    else:
        hwp_qbk = None

    enumeration = list(generate_high_low_enum(lattice_size))
    normalized_term_coefficient = potential_coefficient / (4 * norm)

    interaction_data = InteractionTermData(enumeration, evolution_time, normalized_term_coefficient)
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    interaction_trotter_step.compute(psi_reg, interaction_data)

    if n_batches:
        qc.release_all_rotation_catalyst_qubits()

    metrics = resource_estimator(qc).resources(expanded=True)
    metrics["utilized_qubit_highwater"] = qc.utilized_qubit_highwater
    return metrics


@pytest.mark.parametrize("lattice_size", [8, 10])
def test_interaction_term_qres_with_batched_hwp(lattice_size):
    """Run sanity checks on using one or two batches. Then,
    cross-verify against results in arxiv:2411.02160.
    """
    metrics_no_hwp = setup_interaction_circuits_for_hwp_qre_testing(n_batches=None, lattice_size=lattice_size)
    metrics_one_batch = setup_interaction_circuits_for_hwp_qre_testing(n_batches=1, lattice_size=lattice_size)
    metrics_two_batch = setup_interaction_circuits_for_hwp_qre_testing(n_batches=2, lattice_size=lattice_size)

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
    assert np.isclose(nTof, metrics_two_batch["aggregated_toff_count"])
