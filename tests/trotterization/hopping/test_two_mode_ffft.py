"""Tests for the two mode FFFT qubrick."""

import numpy as np
import psiqdk.workbench.opcodes as opc
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.experimental.active_volume_estimation import op_av_lookup_table
from psiqdk.workbench.ops import QPU_op
from psiqdk.workbench.utility_filters import UnitaryMatrixFilter
from psiqdk.workbench.utils.numpy_utils import fidelity, reverse_numpy_op

from psiq_fh.trotterization.hopping import TwoModeFFFTViaControlledHad, TwoModeFFFTViaPPRs


def compare_arrays_global_phase(A, B, rtol=1e-05, atol=1e-08):
    """Compare two 2D numpy arrays up to a global phase factor.

    Args:
        A (numpy.ndarray): The first 2D numpy array.
        B (numpy.ndarray): The second 2D numpy array.
        rtol (float, optional): Relative tolerance for comparison. Defaults to 1e-05.
        atol (float, optional): Absolute tolerance for comparison. Defaults to 1e-08.

    Returns:
        bool: True if the arrays are equal up to a global phase, False otherwise.
    """
    if A.shape != B.shape:
        return False

    if np.all(np.isclose(A, 0, atol=atol)) or np.all(np.isclose(B, 0, atol=atol)):
        return np.all(np.isclose(A, B, rtol=rtol, atol=atol))

    phase_ratios = B[~np.isclose(A, 0, atol=atol)] / A[~np.isclose(A, 0, atol=atol)]

    phase_factor = phase_ratios[0] / np.abs(phase_ratios[0])

    return np.all(np.isclose(A * phase_factor, B, rtol=rtol, atol=atol))


def test_two_mode_ffft_has_desired_action_on_two_qubits():
    qc = QPU()
    qc.reset(2)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaControlledHad()
    two_mode_ffft.compute(one, two)

    final_state = qc.pull_state()

    expected_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 0, 0, -1],
        ]
    )

    assert np.allclose(final_state, expected_matrix @ initial)

    # Check unitary
    UNITARY = UnitaryMatrixFilter()

    unitary_filter = UNITARY
    qc = QPU(pre_filters=[">>witness>>"], filters=[unitary_filter, ">>buffer>>"])
    qc.reset(2)
    UNITARY.qc = qc
    UNITARY.clear()

    i = Qubits(1, "i", qc)
    j = Qubits(1, "j", qc)

    two_mode_ffft = TwoModeFFFTViaControlledHad()
    two_mode_ffft.compute(j, i)

    gate = reverse_numpy_op(UNITARY.get())

    assert np.allclose(gate, expected_matrix)


def test_two_mode_ffft_v3_has_desired_action_on_two_qubits():
    qc = QPU()
    qc.reset(2)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaPPRs()
    two_mode_ffft.compute(one, two)

    final_state = qc.pull_state()

    expected_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 0, 0, -1],
        ]
    )

    assert np.isclose(fidelity(final_state, expected_matrix @ initial), 1)

    # Check unitary
    UNITARY = UnitaryMatrixFilter()

    unitary_filter = UNITARY
    qc = QPU(pre_filters=[">>witness>>"], filters=[unitary_filter, ">>buffer>>"])
    qc.reset(2)
    UNITARY.qc = qc
    UNITARY.clear()

    i = Qubits(1, "i", qc)
    j = Qubits(1, "j", qc)

    two_mode_ffft = TwoModeFFFTViaPPRs()
    two_mode_ffft.compute(j, i)

    gate = reverse_numpy_op(UNITARY.get())

    assert compare_arrays_global_phase(gate, expected_matrix, rtol=1e-15, atol=1e-15)


def test_two_mode_ffft_has_desired_action_on_two_qubits_in_larger_hilbert_space():
    """Test action of two-mode FFFT operator on a random initial state of a pair of qubits
    in a larger register.
    """
    qc = QPU()
    qc.reset(4)

    qubs = Qubits(4, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaControlledHad()
    two_mode_ffft.compute(qubs[1], qubs[2])

    final_state = qc.pull_state()

    expected_matrix = np.kron(
        np.kron(
            np.eye(2),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, 0, 0, -1],
                ]
            ),
        ),
        np.eye(2),
    )

    assert np.allclose(final_state, expected_matrix @ initial)


def test_two_mode_ffft_v3_has_desired_action_on_two_qubits_in_larger_hilbert_space():
    """Test action of two-mode FFFT operator (decomposed into PPRs) on a random initial state of
    a pair of qubits in a larger register.
    """
    qc = QPU()
    qc.reset(4)

    qubs = Qubits(4, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaPPRs()
    two_mode_ffft.compute(qubs[1], qubs[2])

    final_state = qc.pull_state()

    expected_matrix = np.kron(
        np.kron(
            np.eye(2),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, 0, 0, -1],
                ]
            ),
        ),
        np.eye(2),
    )

    # equivalent up to a global phase in the PPR decomposition
    assert np.isclose(fidelity(final_state, expected_matrix @ initial), 1)


def test_two_mode_ffft_does_nothing_when_control_is_off():
    qc = QPU()
    qc.reset(3)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)
    control = Qubits(1, "ctrl", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    control.write(0)
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaControlledHad()
    two_mode_ffft.compute(one, two, control)

    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)


def test_two_mode_ffft_v3_does_nothing_when_control_is_off():
    qc = QPU()
    qc.reset(3)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)
    control = Qubits(1, "ctrl", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    control.write(0)
    initial = qc.pull_state()

    two_mode_ffft = TwoModeFFFTViaPPRs()
    two_mode_ffft.compute(one, two, control)

    final_state = qc.pull_state()

    # equivalent up to a global phase in the PPR decomposition
    assert np.isclose(fidelity(final_state, initial), 1)
