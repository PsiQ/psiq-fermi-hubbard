"""Tests for the hopping term in the Fermi-Hubbard Trotterizaton circuit."""

import numpy as np
import pytest
from openfermion import FermionOperator, get_sparse_operator, jordan_wigner
from psiqdk.algorithms import (
    ComputeHammingWeightGroupOfThrees,
    ComputeHammingWeightNaive,
    PowerOfTwoBatchedHammingWeightPhasing,
)
from psiqdk.algorithms.utils import PauliMask, PauliSum, pauli_sum_to_numpy
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.arithmetic import NaiveAdd
from psiqdk.workbench.qre import resource_estimator
from psiqdk.workbench.utility_filters import UnitaryMatrixFilter
from psiqdk.workbench.utils.numpy_utils import fidelity, reverse_numpy_op
from scipy.linalg import expm

from psiq_fh.trotterization.fermi_hubbard_data import PlaquetteTermData
from psiq_fh.utils.jw_ordering_utils import generate_even_odd_enum
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, exptXXYY, exptXXYYViaPPR


def compute_unitary_of_plaquette_operator():
    """Compute unitary matrix for plaquette operator, corresponding to periodic hopping around square plaquette.

    Notes:
        - Denoted as K in arxiv:2012.09238 where,
        K = a1† a4 + a1† a2 + a2† a1 + a2† a3 + a3† a2 + a3† a4 + a4† a1 + a4† a3 and,
        a_i/a_i† are fermionic ladder operators

    """
    # Qubit operator for fermionic creation operator at qubit 0 padded to act on a 4 qubit system
    a1 = get_sparse_operator(jordan_wigner(FermionOperator("0", 1))).todense()
    a1 = reverse_numpy_op(np.kron(a1, np.eye(8)))

    a2 = get_sparse_operator(jordan_wigner(FermionOperator("1", 1))).todense()
    a2 = reverse_numpy_op(np.kron(a2, np.eye(4)))

    a3 = get_sparse_operator(jordan_wigner(FermionOperator("2", 1))).todense()
    a3 = reverse_numpy_op(np.kron(a3, np.eye(2)))

    a4 = get_sparse_operator(jordan_wigner(FermionOperator("3", 1))).todense()
    a4 = reverse_numpy_op(a4)

    k_operator = (
        (a1.T.conj() @ a4)
        + (a1.T.conj() @ a2)
        + (a2.T.conj() @ a1)
        + (a2.T.conj() @ a3)
        + (a3.T.conj() @ a2)
        + (a3.T.conj() @ a4)
        + (a4.T.conj() @ a1)
        + (a4.T.conj() @ a3)
    )
    return k_operator


def compute_unitary_of_plaq_time_evolution_from_fermionic_ham(evolution_time, coefficient=1):
    """Compute unitary matrix for time evolution under plaquette operator, from
       fermionic operator representation of plaquette operator

    Args:
        evolution_time: tau in arxiv:2012.09238

    Note:
        - Denoted as exp(i tau K) in arxiv:2012.09238 for evolution time tau
    """
    k_operator = compute_unitary_of_plaquette_operator()
    return expm(-1j * evolution_time * coefficient * k_operator)


def compute_unitary_of_plaq_time_evolution_from_qubit_ham(evolution_time, coefficient=1):
    """Compute unitary matrix for time evolution under plaquette operator, from
       qubit operator representation of plaquette operator

    Args:
        evolution_time: tau in arxiv:2012.09238

    Note:
        - Denoted as exp(i tau K) in arxiv:2012.09238 for evolution time tau
    """
    # Jordan-Wigner transformation already applied to K from
    # arxiv:2012.09238
    # 0-1
    # | |
    # 3-2
    plaquette_hamiltonian = PauliSum(
        [-1 / 2, PauliMask.from_pauli_string("X0 X1")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Y1")],
        [-1 / 2, PauliMask.from_pauli_string("X1 X2")],
        [-1 / 2, PauliMask.from_pauli_string("Y1 Y2")],
        [-1 / 2, PauliMask.from_pauli_string("X2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X0 Z1 Z2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Z1 Z2 Y3")],
    )

    plaquette_hamiltonian = pauli_sum_to_numpy(plaquette_hamiltonian)
    time_evolution = reverse_numpy_op(expm(1j * evolution_time * coefficient * plaquette_hamiltonian))
    return time_evolution


def compute_diagonalizing_unitary():
    """Compute unitary matrix for diagonalizing the plaquette operator.

    Note:
        - Denoted as V in arxiv:2012.09238 though the circuits are different
    """
    n_qubits = 4

    # Obtain unitary for part of a plaquette that diagonalizes the operator
    UNITARY = UnitaryMatrixFilter()
    qc = QPU(num_qubits=n_qubits, pre_filters=[UNITARY, ">>buffer>>"])
    UNITARY.qc = qc
    UNITARY.clear()

    qubs = Qubits(n_qubits, "qubs", qc)

    plaquette_trotter_step = PlaquetteTrotterStep()

    plaquette_trotter_step.fermionic_swap_qubrick.compute(qubs, [1], [2], dagger=True)
    plaquette_trotter_step.two_mode_ffft_qubrick.compute(qubs[2], qubs[3], dagger=True)
    plaquette_trotter_step.two_mode_ffft_qubrick.compute(qubs[1], qubs[0], dagger=True)

    # NOTE: Normally, this part is absorbed into the middle rotation term
    # to make it the exp(i(XX+YY)t) term
    plaquette_trotter_step.two_mode_ffft_qubrick.compute(qubs[1], qubs[2], dagger=True)

    V = UNITARY.get()
    return V


def compute_plaquette_unitary(evolution_time, coefficient=1):
    """Compute unitary matrix for evolution under the plaquette operator,
       from Workbench.

    Args:
        evolution_time: tau in arxiv:2012.09238
    """
    n_qubits = 4

    # Obtain unitary for a plaquette operator
    UNITARY = UnitaryMatrixFilter()
    qc = QPU(num_qubits=n_qubits, pre_filters=[UNITARY, ">>buffer>>"])
    UNITARY.qc = qc
    UNITARY.clear()

    qubs = Qubits(n_qubits, "qubs", qc)

    enumeration = [np.array([[0, 3], [1, 2]])]

    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    plaquette_trotter_step = PlaquetteTrotterStep()

    plaquette_trotter_step.compute(qubs, plaq_data)
    plaquette_unitary = UNITARY.get()
    return plaquette_unitary


def is_diagonal_matrix(arr, threshold=1e-15):
    """Returns True if input is a diagonal matrix."""
    # Set numerical zeros to zeros
    arr[np.abs(arr) < threshold] = 0

    # Count number of nonzero elements in off-diagonal
    if np.count_nonzero(arr - np.diag(np.diagonal(arr))) == 0:
        return True
    else:
        return False


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 0.154])
@pytest.mark.parametrize("coefficient", [1, 0.826, 6])
def test_unitaries_for_plaquette_time_evolution(evolution_time, coefficient):
    """Check that all three unitaries (fermionic, qubit, WB) match up."""
    time_evolution_fham = compute_unitary_of_plaq_time_evolution_from_fermionic_ham(evolution_time, coefficient)
    time_evolution_qham = compute_unitary_of_plaq_time_evolution_from_qubit_ham(evolution_time, coefficient)
    assert np.allclose(time_evolution_fham, time_evolution_qham)

    U_plaq = compute_plaquette_unitary(evolution_time, coefficient)
    assert np.allclose(U_plaq, time_evolution_qham)


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 75])
def test_plaquette_operator_is_correctly_diagonalized(evolution_time):
    """Check that the diagonalizing transformation V does in fact
    diagonalize K and expm(i tau K).
    """
    k_op = compute_unitary_of_plaquette_operator()
    V = compute_diagonalizing_unitary()

    k_diagonalized = V @ k_op @ V.T.conj()
    assert is_diagonal_matrix(k_diagonalized, threshold=1e-15)

    # Reference value of RHS of diagonalized operator in (E10) of arxiv:2012.09238
    a2 = get_sparse_operator(jordan_wigner(FermionOperator("1", 1))).todense()
    a2 = reverse_numpy_op(np.kron(a2, np.eye(4)))

    a3 = get_sparse_operator(jordan_wigner(FermionOperator("2", 1))).todense()
    a3 = reverse_numpy_op(np.kron(a3, np.eye(2)))

    assert np.allclose(
        (2 * (a2.T.conj() @ a2 - a3.T.conj() @ a3)), np.diag([0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0])
    )
    assert np.allclose(k_diagonalized, np.diag([0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0]))

    # Consider time evolution under K
    U_plaq = compute_plaquette_unitary(evolution_time)
    U_plaq_diagonalized = V @ U_plaq @ V.T.conj()
    assert is_diagonal_matrix(U_plaq_diagonalized, threshold=1e-15)


@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 75])
@pytest.mark.parametrize("hwp_qbk", [None, PowerOfTwoBatchedHammingWeightPhasing])
@pytest.mark.parametrize("coefficient", [1, 0.43])
def test_plaquette_trotter_step_has_intended_action_on_one_plaquette(evolution_time, hwp_qbk, use_ppr, coefficient):
    """Test that the plaquette Trotter step has the intended action on one plaquette."""
    if hwp_qbk:
        hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    time_evolution = compute_unitary_of_plaq_time_evolution_from_qubit_ham(evolution_time, coefficient=coefficient)

    sys_qubits = 4
    hw_qubits = 6 if hwp_qbk else 0
    n_qubits = sys_qubits + hw_qubits

    qc = QPU(num_qubits=n_qubits)

    qubs = Qubits(sys_qubits, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hwp_qbk:
        qc.write(0, ~(qubs).mask())
    initial = qc.pull_state()

    enumeration = [np.array([[0, 3], [1, 2]])]  # a test enumeration yeilding one plaquette
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    if use_ppr:
        exptXXYY_qubrick = exptXXYYViaPPR(hwp_qbk)
    else:
        exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    plaquette_trotter_step.compute(qubs, plaq_data)

    if hwp_qbk:
        # Handling 0-angle catalyst register release
        if evolution_time == 0:
            plaquette_trotter_step.exptXXYY_qubrick.catalyst.had()
        else:
            qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if hwp_qbk:
        expected_state = time_evolution @ initial[: 2**sys_qubits]
        pad = np.zeros(2 ** (n_qubits - sys_qubits))
        pad[0] = 1
        expected_state = np.kron(pad, expected_state)
    else:
        expected_state = time_evolution @ initial

    if not hwp_qbk:
        assert np.allclose(final_state, expected_state)
    assert np.isclose(fidelity(final_state, expected_state), 1)


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0.36])
def test_controlled_plaquette_trotter_step(evolution_time):
    """Test control functionality for plaquette Trotter step."""
    enumeration = [np.array([[0, 3], [1, 2]])]  # a test enumeration yeilding one plaquette
    coefficient = 0.367
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    # Control qubit is set to 0, nothing applied
    qc = QPU(num_qubits=5)

    qubs = Qubits(4, "qubs", qc)
    control = Qubits(1, "control", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    control.write(0)
    initial = qc.pull_state()

    plaquette_trotter_step = PlaquetteTrotterStep()
    plaquette_trotter_step.compute(qubs, plaq_data, ctrl=control)

    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)

    # Control qubit is set to 1, unitary applied
    qc = QPU(num_qubits=5)

    qubs = Qubits(4, "qubs", qc)
    control = Qubits(1, "control", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    control.write(1)
    initial = qc.pull_state()

    plaquette_trotter_step = PlaquetteTrotterStep()
    plaquette_trotter_step.compute(qubs, plaq_data, ctrl=control)

    final_state = qc.pull_state()

    assert not np.allclose(final_state, initial)


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 0.24])
@pytest.mark.parametrize("coefficient", [1, 0.4])
@pytest.mark.parametrize("hwp_qbk", [None, PowerOfTwoBatchedHammingWeightPhasing])
@pytest.mark.parametrize("use_ppr", [True, False])
def test_plaquette_trotter_step_has_intended_action_on_two_spin_plaquettes(
    evolution_time, coefficient, hwp_qbk, use_ppr
):
    """Test that the plaquette Trotter step has the intended action on two spin plaquettes."""
    if hwp_qbk:
        hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    # A plaquette Hamiltonian for two plaquettes
    # 0-3
    # | |
    # 1-2
    #
    # 4-7
    # | |
    # 5-6
    plaquette_hamiltonian = PauliSum(
        [-1 / 2, PauliMask.from_pauli_string("X0 X1")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Y1")],
        [-1 / 2, PauliMask.from_pauli_string("X1 X2")],
        [-1 / 2, PauliMask.from_pauli_string("Y1 Y2")],
        [-1 / 2, PauliMask.from_pauli_string("X2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X0 Z1 Z2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Z1 Z2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X4 X5")],
        [-1 / 2, PauliMask.from_pauli_string("Y4 Y5")],
        [-1 / 2, PauliMask.from_pauli_string("X5 X6")],
        [-1 / 2, PauliMask.from_pauli_string("Y5 Y6")],
        [-1 / 2, PauliMask.from_pauli_string("X6 X7")],
        [-1 / 2, PauliMask.from_pauli_string("Y6 Y7")],
        [-1 / 2, PauliMask.from_pauli_string("X4 Z5 Z6 X7")],
        [-1 / 2, PauliMask.from_pauli_string("Y4 Z5 Z6 Y7")],
    )
    plaquette_hamiltonian = coefficient * pauli_sum_to_numpy(plaquette_hamiltonian)

    time_evolution = reverse_numpy_op(expm(1j * evolution_time * plaquette_hamiltonian))

    sys_qubits = 8
    hw_qubits = 9 if hwp_qbk else 0
    n_qubits = sys_qubits + hw_qubits

    qc = QPU(num_qubits=n_qubits)

    qubs = Qubits(sys_qubits, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hwp_qbk:
        qc.write(0, ~(qubs).mask())
    initial = qc.pull_state()

    enumeration = [np.array([[0, 3], [1, 2]]), np.array([[4, 7], [5, 6]])]
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    if use_ppr:
        exptXXYY_qubrick = exptXXYYViaPPR(hwp_qbk)
    else:
        exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    plaquette_trotter_step.compute(qubs, plaq_data)

    if hwp_qbk:
        # Handling 0-angle catalyst register release
        if evolution_time == 0:
            plaquette_trotter_step.exptXXYY_qubrick.catalyst.had()
        else:
            qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if hwp_qbk:
        expected_state = time_evolution @ initial[: 2**sys_qubits]
        pad = np.zeros(2 ** (n_qubits - sys_qubits))
        pad[0] = 1
        expected_state = np.kron(pad, expected_state)
    else:
        expected_state = time_evolution @ initial

    if not hwp_qbk:
        assert np.allclose(final_state, expected_state)
    assert np.isclose(fidelity(final_state, expected_state), 1)


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 0.683])
@pytest.mark.parametrize("coefficient", [1, 0.36])
@pytest.mark.parametrize("hwp_qbk", [None, PowerOfTwoBatchedHammingWeightPhasing])
def test_plaquette_trotter_step_has_intended_action_with_nonlocal_fermionic_swap(evolution_time, coefficient, hwp_qbk):
    """Test that the plaquette Trotter step has the intended action with nonlocal fermionic swaps."""
    if hwp_qbk:
        hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    # a plaquette Hamiltonian where non-local fswaps are needed
    # 0-5
    # | |
    # 1-2
    plaquette_hamiltonian = PauliSum(
        [-1 / 2, PauliMask.from_pauli_string("X0 X1")],  # 0, 1, 2, 5
        [-1 / 2, PauliMask.from_pauli_string("Y0 Y1")],
        [-1 / 2, PauliMask.from_pauli_string("X1 X2")],
        [-1 / 2, PauliMask.from_pauli_string("Y1 Y2")],
        [-1 / 2, PauliMask.from_pauli_string("X2 Z3 Z4 X5")],
        [-1 / 2, PauliMask.from_pauli_string("Y2 Z3 Z4 Y5")],
        [-1 / 2, PauliMask.from_pauli_string("X0 Z1 Z2 Z3 Z4 X5")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Z1 Z2 Z3 Z4 Y5")],
    )

    plaquette_hamiltonian = coefficient * pauli_sum_to_numpy(plaquette_hamiltonian)

    time_evolution = reverse_numpy_op(expm(1j * evolution_time * (plaquette_hamiltonian)))

    sys_qubits = 6
    hw_qubits = 7 if hwp_qbk else 0
    n_qubits = sys_qubits + hw_qubits
    qc = QPU(num_qubits=n_qubits)

    qubs = Qubits(sys_qubits, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hwp_qbk:
        qc.write(0, ~(qubs).mask())
    initial = qc.pull_state()

    enumeration = [np.array([[0, 5], [1, 2]])]
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    plaquette_trotter_step.compute(qubs, plaq_data)

    if hwp_qbk:
        # Handling 0-angle catalyst register release
        if evolution_time == 0:
            plaquette_trotter_step.exptXXYY_qubrick.catalyst.had()
        else:
            qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if hwp_qbk:
        expected_state = time_evolution @ initial[: 2**sys_qubits]
        pad = np.zeros(2 ** (n_qubits - sys_qubits))
        pad[0] = 1
        expected_state = np.kron(pad, expected_state)
    else:
        expected_state = time_evolution @ initial

    if not hwp_qbk:
        assert np.allclose(final_state, expected_state)
    assert np.isclose(fidelity(final_state, expected_state), 1)


def setup_plaquette_hams_for_2x4_lattice():
    """Returns plaquette hamiltonians for 2x4 lattice but
       corresponding to one of the spins.
       Pink: [[0, 1, 3, 2], [4, 5, 7, 6]]

       0-2-4-6
       | | | |
       1-3-5-7

       Gold: [[3, 2, 4, 5], [7, 6, 0, 1]]

    Note:
        - Explains use of 8 qubits instead of 16
        - Not following the qubit enumeration scheme from spinful case
        - The gold plaquettes are obtained by first shift the enumeration up and left so it will look like:
            3-5-7-1
            | | | |
            2-4-6-0
            and then get the plaquettes by starting top left and then going down, right, up.
    """
    pink_plaquette_hamiltonian = PauliSum(
        [-1 / 2, PauliMask.from_pauli_string("X0 X1")],  # 0, 1, 3, 2
        [-1 / 2, PauliMask.from_pauli_string("Y0 Y1")],
        [-1 / 2, PauliMask.from_pauli_string("X1 Z2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y1 Z2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X2 X3")],
        [-1 / 2, PauliMask.from_pauli_string("Y2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X0 Z1 X2")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Z1 Y2")],
        [-1 / 2, PauliMask.from_pauli_string("X4 X5")],  # 4, 5, 7, 6
        [-1 / 2, PauliMask.from_pauli_string("Y4 Y5")],
        [-1 / 2, PauliMask.from_pauli_string("X5 Z6 X7")],
        [-1 / 2, PauliMask.from_pauli_string("Y5 Z6 Y7")],
        [-1 / 2, PauliMask.from_pauli_string("X6 X7")],
        [-1 / 2, PauliMask.from_pauli_string("Y6 Y7")],
        [-1 / 2, PauliMask.from_pauli_string("X4 Z5 X6")],
        [-1 / 2, PauliMask.from_pauli_string("Y4 Z5 Y6")],
    )

    gold_plaquette_hamiltonian = PauliSum(
        [-1 / 2, PauliMask.from_pauli_string("X2 X3")],  # 3, 2, 4, 5
        [-1 / 2, PauliMask.from_pauli_string("Y2 Y3")],
        [-1 / 2, PauliMask.from_pauli_string("X2 Z3 X4")],
        [-1 / 2, PauliMask.from_pauli_string("Y2 Z3 Y4")],
        [-1 / 2, PauliMask.from_pauli_string("X4 X5")],
        [-1 / 2, PauliMask.from_pauli_string("Y4 Y5")],
        [-1 / 2, PauliMask.from_pauli_string("X3 Z4 X5")],
        [-1 / 2, PauliMask.from_pauli_string("Y3 Z4 Y5")],
        [-1 / 2, PauliMask.from_pauli_string("X6 X7")],  # 7, 6, 0, 1
        [-1 / 2, PauliMask.from_pauli_string("Y6 Y7")],
        [-1 / 2, PauliMask.from_pauli_string("X0 Z1 Z2 Z3 Z4 Z5 X6")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Z1 Z2 Z3 Z4 Z5 Y6")],
        [-1 / 2, PauliMask.from_pauli_string("X0 X1")],
        [-1 / 2, PauliMask.from_pauli_string("Y0 Y1")],
        [-1 / 2, PauliMask.from_pauli_string("X1 Z2 Z3 Z4 Z5 Z6 X7")],
        [-1 / 2, PauliMask.from_pauli_string("Y1 Z2 Z3 Z4 Z5 Z6 Y7")],
    )
    return pink_plaquette_hamiltonian, gold_plaquette_hamiltonian


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0, 0.386])
@pytest.mark.parametrize("coefficient", [1, 0.3])
def test_plaquette_trotter_steps_have_intended_action_on_pink_and_gold_plaquettes_for_2_x_4_lattice(
    evolution_time, coefficient
):
    """Test that the plaquette Trotter steps have the intended action on pink and gold plaquettes for 2x4 lattice."""
    (pink_plaquette_hamiltonian, gold_plaquette_hamiltonian) = setup_plaquette_hams_for_2x4_lattice()

    # Test pink plaquette action
    pink_time_evolution = reverse_numpy_op(
        expm(1j * evolution_time * coefficient * pauli_sum_to_numpy(pink_plaquette_hamiltonian))
    )

    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()
    enumeration = [np.array([[0, 2, 4, 6], [1, 3, 5, 7]])]
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    pink_plaquette_trotter_step = PlaquetteTrotterStep()
    pink_plaquette_trotter_step.compute(qubs, plaq_data)

    final_state = qc.pull_state()

    assert np.allclose(final_state, pink_time_evolution @ initial)

    # Test gold plaquette action
    gold_time_evolution = reverse_numpy_op(
        expm(1j * evolution_time * coefficient * pauli_sum_to_numpy(gold_plaquette_hamiltonian))
    )

    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    plaq_data = PlaquetteTermData("gold", enumeration, evolution_time, coefficient)

    gold_plaquette_trotter_step = PlaquetteTrotterStep()
    gold_plaquette_trotter_step.compute(qubs, plaq_data)

    final_state = qc.pull_state()

    assert np.allclose(final_state, gold_time_evolution @ initial)


@pytest.mark.parametrize("evolution_time", [-50, -1.783, 0.4])
@pytest.mark.parametrize("coefficient", [1, 0.46])
def test_plaquette_trotter_steps_are_invertible(evolution_time, coefficient):
    """Test that the plaquette Trotter steps are invertible."""
    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    enumeration = [np.array([[0, 2, 4, 6], [1, 3, 5, 7]])]
    plaq_data = PlaquetteTermData("gold", enumeration, evolution_time, coefficient)

    gold_plaquette_trotter_step = PlaquetteTrotterStep()
    gold_plaquette_trotter_step.compute(qubs, plaq_data)

    # Instead of negating the angle, you can modify the evolution time to its
    # negative
    plaq_data.evolution_time *= -1
    gold_plaquette_trotter_step.compute(qubs, plaq_data)

    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)


@pytest.mark.parametrize("evolution_time", [50, 0.783])
@pytest.mark.parametrize("coefficient", [1, 0.81])
def test_sequential_plaquette_trotter_steps_build_up(evolution_time, coefficient):
    """Test that sequential plaquette Trotter steps compose to the intended action."""
    (_, gold_plaquette_hamiltonian) = setup_plaquette_hams_for_2x4_lattice()

    gold_time_evolution = reverse_numpy_op(
        expm(1j * 2 * evolution_time * (coefficient * pauli_sum_to_numpy(gold_plaquette_hamiltonian)))
    )

    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    enumeration = [np.array([[0, 2, 4, 6], [1, 3, 5, 7]])]  # 2x4 lattice
    plaq_data = PlaquetteTermData("gold", enumeration, evolution_time, coefficient)

    gold_plaquette_trotter_step = PlaquetteTrotterStep()
    gold_plaquette_trotter_step.compute(qubs, plaq_data)
    gold_plaquette_trotter_step.compute(qubs, plaq_data)

    final_state = qc.pull_state()

    assert np.allclose(final_state, gold_time_evolution @ initial)


@pytest.mark.parametrize("evolution_time", [-50, 1.783])
@pytest.mark.parametrize("coefficient", [1, -1, 0.25])
@pytest.mark.parametrize("hwp_qbk", [None, PowerOfTwoBatchedHammingWeightPhasing])
def test_plaquette_trotter_steps_of_pink_and_gold_reproduce_kinetic_evolution_for_2_by_4_lattice(
    evolution_time, coefficient, hwp_qbk
):
    """Tests 2x4 lattice but corresponding to one of the spins.

    Note:
        - Explains use of 8 qubits instead of 16
    """
    if hwp_qbk:
        hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    if evolution_time == 0 and hwp_qbk is not None:
        pytest.skip("Catalyst for zero_angle not handled yet.")

    (pink_plaquette_hamiltonian, gold_plaquette_hamiltonian) = setup_plaquette_hams_for_2x4_lattice()
    kinetic_hamiltonian = coefficient * (pink_plaquette_hamiltonian + gold_plaquette_hamiltonian)

    kinetic_time_evolution = reverse_numpy_op(expm(1j * evolution_time * (pauli_sum_to_numpy(kinetic_hamiltonian))))

    sys_qubits = 8
    hw_qubits = 9 if hwp_qbk else 0
    n_qubits = sys_qubits + hw_qubits
    qc = QPU(num_qubits=n_qubits)

    qubs = Qubits(sys_qubits, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    if hwp_qbk:
        qc.write(0, ~(qubs).mask())
    initial = qc.pull_state()

    enumeration = [np.array([[0, 2, 4, 6], [1, 3, 5, 7]])]  # 2x4 lattice

    exptXXYY_qubrick = exptXXYY(hwp_qbk)

    pink_plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    gold_plaq_data = PlaquetteTermData("gold", enumeration, evolution_time, coefficient)
    gold_plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    # No plaquette commutator error with 2x4, so don't need to split up steps
    pink_plaquette_trotter_step.compute(qubs, pink_plaq_data)
    gold_plaquette_trotter_step.compute(qubs, gold_plaq_data)

    if hwp_qbk:
        qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if hwp_qbk:
        expected_state = kinetic_time_evolution @ initial[: 2**sys_qubits]
        pad = np.zeros(2 ** (n_qubits - sys_qubits))
        pad[0] = 1
        expected_state = np.kron(pad, expected_state)
    else:
        expected_state = kinetic_time_evolution @ initial

    if not hwp_qbk:
        assert np.allclose(final_state, expected_state)
    assert np.isclose(fidelity(final_state, expected_state), 1)


def test_plaquette_step_uncompute_with_hamming_weight_phasing():
    """Test that uncompute with HWP works correctly.

    Note:
        - This passes if catalyst qubits are not released
          in between compute and uncompute. Or else it fails.
          If released, qc should allocate the right catalyst
          when uncompute is called.
    """
    evolution_time = -50

    qc = QPU(num_qubits=10)

    qubs = Qubits(4, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    qc.write(0, ~(qubs).mask())
    initial = qc.pull_state()

    # Hamming weight qubrick
    hw_qbk = ComputeHammingWeightNaive(adder=NaiveAdd())
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hw_qbk)

    enumeration = [np.array([[0, 3], [1, 2]])]
    coefficient = 1.3
    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    qc.label("compute")
    plaquette_trotter_step.compute(qubs, plaq_data)
    qc.label()

    qc.label("uncompute")
    plaquette_trotter_step.uncompute()
    qc.label()
    qc.release_all_rotation_catalyst_qubits()

    uncomputed_state = qc.pull_state()

    assert np.allclose(initial, uncomputed_state)


def setup_plaquette_circuits_for_hwp_wf_testing(n_batches, init_state):
    """Return state vector and metrics from running plaquette circuit
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
        num_qubits=total_num_qubits,
        pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"],
    )

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    qc.push_state(init_state)

    evolution_time = np.pi

    if n_batches:
        hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
        hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=n_batches)
    else:
        hwp_qbk = None

    coefficient = 0.76

    # Only pink plaquette in 2x2
    enumeration = list(generate_even_odd_enum(lattice_size))

    plaq_data = PlaquetteTermData("pink", enumeration, evolution_time, coefficient)

    exptXXYY_qubrick = exptXXYY(hwp_qbk)
    plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qubrick)

    plaquette_trotter_step.compute(psi_reg, plaq_data)

    if n_batches:
        qc.release_all_rotation_catalyst_qubits()

    metrics = resource_estimator(qc).resources(expanded=True)
    metrics["utilized_qubit_highwater"] = qc.utilized_qubit_highwater

    wf = qc.pull_state()
    return wf, metrics


def test_2x2_plaquette_term_with_batched_hwp():
    """Test 2x2 plaquette terms with batched HWP."""
    # Set random state
    lattice_size = 2
    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites * 2 + 4
    qc = QPU(
        num_qubits=total_num_qubits, pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>witness>>"]
    )
    psi_reg = Qubits(number_spin_sites, "psi", qc)
    qc.set_random()
    qc.write(0, ~(psi_reg).mask())
    init_state = qc.pull_state()

    wf_no_hwp, metrics_no_hwp = setup_plaquette_circuits_for_hwp_wf_testing(n_batches=None, init_state=init_state)
    wf_one_batch, metrics_one_batch = setup_plaquette_circuits_for_hwp_wf_testing(n_batches=1, init_state=init_state)
    wf_two_batch, metrics_two_batch = setup_plaquette_circuits_for_hwp_wf_testing(n_batches=2, init_state=init_state)

    assert np.isclose(fidelity(wf_no_hwp, wf_one_batch), 1)
    assert np.isclose(fidelity(wf_no_hwp, wf_two_batch), 1)

    # Check that the circuits are different
    assert metrics_no_hwp["utilized_qubit_highwater"] != metrics_one_batch["utilized_qubit_highwater"]
    assert metrics_no_hwp["utilized_qubit_highwater"] != metrics_two_batch["utilized_qubit_highwater"]


def test_error_when_color_incorrect():
    """Test that an error is raised when the color type is incorrect.

    When generating the plaquette qubrick the color type must be pink or gold.
    If it differs from this an error should be raised.
    """
    enumeration = [np.array([[0, 3], [1, 2]])]
    evolution_time = 1
    kinetic_coefficient = 1
    color = "orange"
    with pytest.raises(ValueError, match=r"Invalid color 'orange'; must be 'pink' or 'gold'."):
        PlaquetteTermData(color, enumeration, evolution_time, kinetic_coefficient)
