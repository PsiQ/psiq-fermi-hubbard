"""Tests for the interaction term in the Fermi-Hubbard Trotterization circuit in arxiv:2012.09238."""

import numpy as np
import pytest
from psiqdk.algorithms import (
    ComputeHammingWeightGroupOfThrees,
    ComputeHammingWeightNaive,
    PowerOfTwoBatchedHammingWeightPhasing,
)
from psiqdk.algorithms.utils.paulimask import PauliMask, PauliSum, pauli_sum_to_numpy
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.arithmetic import NaiveAdd
from psiqdk.workbench.qre import resource_estimator
from psiqdk.workbench.utility_filters import UnitaryMatrixFilter
from psiqdk.workbench.utils.numpy_utils import fidelity, reverse_numpy_op
from scipy.linalg import expm

from psiq_fh.trotterization.fermi_hubbard_data import InteractionTermData
from psiq_fh.utils.jw_ordering_utils import generate_even_odd_enum
from psiq_fh.trotterization.interaction import InteractionTrotterStep


@pytest.mark.parametrize("total_evolution_time", [-526, -16.66677, 178.2, 300])
@pytest.mark.parametrize("potential_coefficient", [-0.23, 0.6])
def test_interaction_unitary_is_exp_itzz(total_evolution_time, potential_coefficient):
    hamiltonian = reverse_numpy_op(pauli_sum_to_numpy(PauliSum([potential_coefficient, PauliMask(0, 3)])))
    time_evolution = expm(1j * total_evolution_time * hamiltonian)

    UNITARY = UnitaryMatrixFilter()
    qc = QPU(filters=[UNITARY])
    qc.reset(2)
    UNITARY.qc = qc
    UNITARY.clear()

    psi_reg = Qubits(2, "psi", qc)

    # instantiate Plaquette Trotterization Qubrick
    # Corresponds to interaction indices [[0, 1]]
    enumeration = np.array([[[0]], [[1]]])
    interaction_data = InteractionTermData(enumeration, total_evolution_time, potential_coefficient)
    interaction_trotter = InteractionTrotterStep()

    # execute trotter step
    interaction_trotter.compute(psi_reg, interaction_data)

    unitary = UNITARY.get()

    assert np.allclose(unitary, time_evolution)


@pytest.mark.parametrize("total_evolution_time", [-526, -16.66677, 178.2, 300])
@pytest.mark.parametrize("potential_coefficient", [-0.23, 0.2])
def test_interaction_unitary_is_exp_itzz_for_multiple_terms(total_evolution_time, potential_coefficient):
    hamiltonian = reverse_numpy_op(
        pauli_sum_to_numpy(
            PauliSum(
                [potential_coefficient / 4, PauliMask.from_pauli_string("Z0 Z1")],
                [potential_coefficient / 4, PauliMask.from_pauli_string("Z2 Z3")],
                [potential_coefficient / 4, PauliMask.from_pauli_string("Z4 Z5")],
                [potential_coefficient / 4, PauliMask.from_pauli_string("Z6 Z7")],
            )
        )
    )
    time_evolution = expm(1j * total_evolution_time * hamiltonian)

    UNITARY = UnitaryMatrixFilter()
    qc = QPU(filters=[UNITARY])
    qc.reset(8)
    UNITARY.qc = qc
    UNITARY.clear()

    psi_reg = Qubits(8, "psi", qc)

    enumeration = generate_even_odd_enum(2)
    interaction_data = InteractionTermData(
        enumeration, total_evolution_time, potential_coefficient / 4
    )  # Factor of 1/4 from Jordan Wigner transform
    interaction_trotter = InteractionTrotterStep()

    # execute trotter step
    interaction_trotter.compute(psi_reg, interaction_data)

    unitary = UNITARY.get()

    assert np.allclose(unitary, time_evolution)


@pytest.mark.parametrize("total_evolution_time", [-526, -16.66677, 178.2, 300])
@pytest.mark.parametrize("potential_coefficient", [-0.23, 0.6])
def test_interaction_step_does_nothing_when_control_is_zero(total_evolution_time, potential_coefficient):
    UNITARY = UnitaryMatrixFilter()
    qc = QPU(filters=[UNITARY])
    qc.reset(3)
    UNITARY.qc = qc
    UNITARY.clear()

    ctrl = Qubits(1, "ctrl", qc)
    psi_reg = Qubits(2, "psi", qc)

    # instantiate Plaquette Trotterization Qubrick
    # Corresponds to interaction indices [[0, 1]]
    enumeration = np.array([[[0]], [[1]]])
    interaction_data = InteractionTermData(enumeration, total_evolution_time, potential_coefficient)
    interaction_trotter = InteractionTrotterStep()

    # execute trotter step
    interaction_trotter.compute(psi_reg, interaction_data, ctrl=ctrl)

    unitary_action_on_psi = reverse_numpy_op(UNITARY.get())[: 1 << len(psi_reg), : 1 << len(psi_reg)]

    assert np.allclose(unitary_action_on_psi, np.eye(1 << (len(psi_reg))))


@pytest.mark.parametrize("total_evolution_time", [-16.66677])
@pytest.mark.parametrize("potential_coefficient", [0.6])
@pytest.mark.parametrize("hw_qbk", [ComputeHammingWeightNaive])
def test_interaction_is_exp_itzz_for_multiple_terms_with_hwp(total_evolution_time, potential_coefficient, hw_qbk):
    """Test interaction term with Hamming weight phasing. Testing the uncontrolled version
       because it's not clear how to pad the time_evolution matrix to account for the control.

    Notes:
        - For controlled, see `test_interaction_unitary_controlled`
        - e^{iHt} convention is used for time-evolution operator rather than e^{−iHt}.
    """
    if hw_qbk:
        hamming_weight_qubrick = hw_qbk(adder=NaiveAdd())
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=1)
    else:
        hwp_qbk = None

    hamiltonian = reverse_numpy_op(
        pauli_sum_to_numpy(
            PauliSum(
                [potential_coefficient, PauliMask.from_pauli_string("Z0 Z1")],
                [potential_coefficient, PauliMask.from_pauli_string("Z2 Z3")],
                [potential_coefficient, PauliMask.from_pauli_string("Z4 Z5")],
                [potential_coefficient, PauliMask.from_pauli_string("Z6 Z7")],
            )
        )
    )
    time_evolution = expm(1j * total_evolution_time * hamiltonian)

    qc = QPU()
    sys_qubits = 8
    hw_qubits = 9 if hw_qbk else 0
    n_qubits = sys_qubits + hw_qubits
    qc.reset(n_qubits)

    psi_reg = Qubits(sys_qubits, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hw_qbk:
        qc.write(0, ~(psi_reg).mask())
    initial = qc.pull_state()

    # instantiate Plaquette Trotterization Qubrick
    enumeration = generate_even_odd_enum(2)
    interaction_data = InteractionTermData(enumeration, total_evolution_time, potential_coefficient)
    interaction_trotter = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    # execute trotter step
    interaction_trotter.compute(psi_reg, interaction_data)

    if hw_qbk:
        qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if hw_qbk:
        expected_state = time_evolution @ initial[: 2**sys_qubits]
        pad = np.zeros(2 ** (n_qubits - sys_qubits))
        pad[0] = 1
        expected_state = np.kron(pad, expected_state)
    else:
        expected_state = reverse_numpy_op(time_evolution) @ initial

    if not hw_qbk:
        assert np.allclose(final_state, expected_state)

    assert np.isclose(fidelity(final_state, expected_state), 1)


@pytest.mark.parametrize("total_evolution_time", [-16.66677])
@pytest.mark.parametrize("potential_coefficient", [0.6])
@pytest.mark.parametrize("particle_hole_symmetry", [1, 0])
def test_interaction_unitary_controlled(total_evolution_time, potential_coefficient, particle_hole_symmetry):
    """Test controlled interaction unitary with and without HWP."""
    # No HWP
    qc = QPU()
    n_qubits = 8
    n_qubits += 1  # ctrl
    n_qubits += 9  # pad
    if not particle_hole_symmetry:
        n_qubits += 6
    qc.reset(n_qubits)

    ctrl = Qubits(1, "ctrl", qc)
    psi_reg = Qubits(8, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    qc.write(0, ~(psi_reg | ctrl).mask())
    initial = qc.pull_state()

    enumeration = generate_even_odd_enum(2)
    interaction_data = InteractionTermData(
        enumeration, total_evolution_time, potential_coefficient, particle_hole_symmetry=particle_hole_symmetry
    )
    interaction_trotter = InteractionTrotterStep()

    interaction_trotter.compute(psi_reg, interaction_data, ctrl)

    final_state = qc.pull_state()

    # HWP
    qc = QPU()
    n_qubits = 8
    n_qubits += 1  # ctrl
    n_qubits += 9  # HWP
    if not particle_hole_symmetry:
        n_qubits += 6
    qc.reset(n_qubits)

    ctrl = Qubits(1, "ctrl", qc)
    psi_reg = Qubits(8, "psi", qc)

    qc.push_state(initial)

    hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    interaction_trotter_hwp = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    interaction_trotter_hwp.compute(psi_reg, interaction_data, ctrl)

    qc.release_all_rotation_catalyst_qubits()

    final_state_hwp = qc.pull_state()

    assert np.allclose(final_state, final_state_hwp)
    assert np.isclose(fidelity(final_state, final_state_hwp), 1)


def test_interaction_step_uncompute_with_hamming_weight_phasing():
    """Test uncompute for interaction step with HWP.

    Note:
        - This passes if catalyst qubits are not released
          in between compute and uncompute. Or else it fails.
          If released, qc should allocate the right catalyst
          when uncompute is called.
    """
    total_evolution_time = -16.666667
    potential_coefficient = 0.6
    hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    qc = QPU()
    sys_qubits = 8
    hw_qubits = 9 if hw_qbk else 0
    n_qubits = sys_qubits + hw_qubits
    qc.reset(n_qubits)

    psi_reg = Qubits(sys_qubits, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hw_qbk:
        qc.write(0, ~(psi_reg).mask())
    initial = qc.pull_state()

    # instantiate Interaction Trotterization Qubrick
    enumeration = generate_even_odd_enum(2)

    interaction_data = InteractionTermData(enumeration, total_evolution_time, potential_coefficient)
    interaction_trotter = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    # execute trotter step
    interaction_trotter.compute(psi_reg, interaction_data)

    interaction_trotter.uncompute()

    qc.release_all_rotation_catalyst_qubits()
    uncomputed_state = qc.pull_state()

    assert np.allclose(initial, uncomputed_state)


def setup_interaction_circuits_for_hwp_wf_testing(n_batches, init_state):
    """Return state vector and metrics from running interaction circuit
    with or without batched hamming weight phasing, for 2x2.

    Args:
         n_batches (int | None): Number of batches of rotations. If None,
             no HWP used.
         init_state (numpy.array): Initial state
    """
    lattice_size = 2

    # set-up total number of qubits:
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 2 + 4

    # Set up QPU instance:
    qc = QPU(
        pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"],
    )
    qc.reset(total_num_qubits)

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    qc.push_state(init_state)

    evolution_time = np.pi
    potential_coefficient = 8

    if n_batches:
        hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=n_batches)
    else:
        hwp_qbk = None

    enumeration = list(generate_even_odd_enum(lattice_size))
    interaction_data = InteractionTermData(enumeration, evolution_time, potential_coefficient)
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    interaction_trotter_step.compute(psi_reg, interaction_data)

    if n_batches:
        qc.release_all_rotation_catalyst_qubits()

    metrics = resource_estimator(qc).resources(expanded=True)
    metrics["utilized_qubit_highwater"] = qc.utilized_qubit_highwater

    wf = qc.pull_state()
    return wf, metrics


def test_2x2_interaction_term_with_batched_hwp():
    """Test 2x2 interaction terms with batched HWP."""
    # Set random state
    lattice_size = 2
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 2 + 4
    qc = QPU(pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"])
    qc.reset(total_num_qubits)
    psi_reg = Qubits(number_spin_sites, "psi", qc)
    qc.set_random()
    qc.write(0, ~(psi_reg).mask())
    init_state = qc.pull_state()

    wf_no_hwp, metrics_no_hwp = setup_interaction_circuits_for_hwp_wf_testing(n_batches=None, init_state=init_state)
    wf_one_batch, metrics_one_batch = setup_interaction_circuits_for_hwp_wf_testing(n_batches=1, init_state=init_state)
    wf_four_batch, metrics_four_batch = setup_interaction_circuits_for_hwp_wf_testing(
        n_batches=4, init_state=init_state
    )

    # global phase difference here
    assert np.isclose(fidelity(wf_no_hwp, wf_one_batch), 1)
    assert np.isclose(fidelity(wf_no_hwp, wf_four_batch), 1)

    # Check that the circuits are different
    assert metrics_no_hwp["utilized_qubit_highwater"] != metrics_one_batch["utilized_qubit_highwater"]
    assert metrics_no_hwp["utilized_qubit_highwater"] != metrics_four_batch["utilized_qubit_highwater"]
