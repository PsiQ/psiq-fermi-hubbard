"""Tests for verifying QREs for a single Trotter step"""

import numpy as np
import pytest
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, PowerOfTwoBatchedHammingWeightPhasing
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.qre import resource_estimator

from psiq_fh.trotterization.catalyst_allocation import fetch_catalyst_qubits
from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.trotterization.fermi_hubbard_trotterization import HubbardPlaquetteTrotterizationIPG
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, exptXXYY
from psiq_fh.trotterization.interaction import InteractionTrotterStep


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
@pytest.mark.parametrize("particle_hole_symmetry", [True])
def test_plaquette_qre(
    lattice_size,
    particle_hole_symmetry,
):
    """Verify QRE for a single Trotter step.

    Note:
        - See Appendix E of 2012.09238, Eq. E1
        - While in the ref above it is claimed that you can merge the first and last term,
        it is unclear that you can do that per step.
        - We assume you make 2 calls to the interaction term. So instead of 4 L^2 rotations,
        we have 5 L^2 rotations total, where 2 L^2 comes from interaction.
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites

    potential_coefficient = 8

    # Set up QPU instance:
    qc = QPU(num_qubits=total_num_qubits, filters=[">>witness>>", ">>buffer>>"])

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    # instantiate Plaquette Trotterization Qubrick
    fh_data = FermiHubbardData(
        x_dim=lattice_size,
        y_dim=lattice_size,
        total_evolution_time=0.4,  # Note: time cannot be np.pi as then rotations are trivial
        n_trotter_steps=1,
        enumeration=None,
        particle_hole_symmetry=particle_hole_symmetry,
        t=1,
        u=potential_coefficient,
    )
    hubbard_unitary = HubbardPlaquetteTrotterizationIPG(
        InteractionTrotterStep(), PlaquetteTrotterStep(), PlaquetteTrotterStep()
    )

    hubbard_unitary.compute(psi_reg, fh_data)

    metrics = qc.metrics()

    assert metrics["rotation_count"] == 5 * lattice_size**2

    # Note: each cHad gate will contribute 2 T-gates
    assert qc.witness.filter("qc.had", condition=1).count() == 6 * lattice_size**2


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
def test_plaquette_qre_with_hwp(
    lattice_size,
):
    """Verify QRE for a single Trotter step using Hamming
       weight phasing.

    Note:
        - See Eq. B26 of 2411.02160 for number of catalyst qubits
    """
    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 5

    number_of_trotter_steps = 1

    # Note: if potential coefficient is 8, the interaction angles and plaquette angles only
    # differ by factors of two and so they can share some of the catalyst state
    # which makes n_rots_cat incorrect
    potential_coefficient = 8.5

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

    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    batched_hamming_weight_qubrick = PowerOfTwoBatchedHammingWeightPhasing(
        hamming_weight_qubrick,
    )

    # Instantiate Plaquette Trotterization Qubrick
    fh_data = FermiHubbardData(
        x_dim=lattice_size,
        y_dim=lattice_size,
        total_evolution_time=0.4,  # Note:time cannot be np.pi as then rotations are trivial,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=None,
        particle_hole_symmetry=True,
        t=1,
        u=potential_coefficient,
    )
    exptXXYY_qbk = exptXXYY(batched_hamming_weight_qubrick)
    hubbard_unitary = HubbardPlaquetteTrotterizationIPG(
        InteractionTrotterStep(batched_hamming_weight_qubrick),
        PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qbk),
        PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qbk),
    )

    hubbard_unitary.compute(psi_reg, fh_data)
    metrics = resource_estimator(qc).resources(expanded=True)

    # Each controlled-Had decomposed into 2 T-gates
    t_count = metrics["t_gates"]
    assert t_count == 12 * lattice_size**2

    # Analytical expression for Toffs for generalized phase gradient for 1 Trotter step
    lsquared = lattice_size**2
    w_lsquared = lsquared.bit_count()
    n_toffs_phase_grad_per_term = lsquared + np.floor(np.log2(lattice_size**2)).astype(int) - np.array(w_lsquared) + 1
    # 5 layers/term calls per step
    n_toffs_phase_grad_all = (4 * number_of_trotter_steps + 1) * n_toffs_phase_grad_per_term
    assert metrics["aggregated_toff_count"] == n_toffs_phase_grad_all

    # Check rotation count
    n_rots_per_term = 4 * number_of_trotter_steps + 1  # in each generalized phase catalysis
    n_rots_cat = 2 * np.floor(np.log2(lattice_size**2)).astype(int) + 3  # See Eq. (B26)
    n_rots = n_rots_per_term + n_rots_cat
    assert np.isclose(metrics["rotations"], n_rots)
    assert fetch_catalyst_qubits(qc).num_qubits == n_rots_cat
