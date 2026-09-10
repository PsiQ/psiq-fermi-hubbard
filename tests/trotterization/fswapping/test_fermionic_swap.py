"""Tests for fermionic swap qubricks."""

import numpy as np
import pytest
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.utility_filters import UnitaryMatrixFilter

from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt


def test_fermionic_swap_has_desired_action():
    """Test that a local/adjacent fermionic swap has the desired action."""
    qc = QPU(num_qubits=2)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    fermionic_swap = FermionicSwapWithReplace()
    fermionic_swap.compute(one | two, [0], [1])

    final_state = qc.pull_state()

    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cz = np.diag([1, 1, 1, -1])

    assert np.allclose(final_state, swap @ cz @ initial), (
        "FermionicSwapWithReplace did not realize CZ then SWAP for a single adjacent pair."
    )  # note: cz comes first in time so on the right here


def test_fermionic_swap_has_desired_action_on_multiple_pairs():
    """Test a set of parallel fermionic swaps has the desired action."""
    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    fermionic_swap = FermionicSwapWithReplace()
    fermionic_swap.compute(qubs, [0, 2, 4, 6], [1, 3, 5, 7])

    final_state = qc.pull_state()

    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cz = np.diag([1, 1, 1, -1])

    # note: cz comes first in time so on the right here
    multi_fswaps = np.kron(swap @ cz, swap @ cz)
    multi_fswaps = np.kron(swap @ cz, multi_fswaps)
    multi_fswaps = np.kron(swap @ cz, multi_fswaps)

    assert np.allclose(final_state, multi_fswaps @ initial)


def test_fermionic_swap_is_invertible():
    """Test that a fermionic swap is invertible."""
    qc = QPU(num_qubits=8)

    qubs = Qubits(8, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    fermionic_swap = FermionicSwapWithReplace()
    fermionic_swap.compute(qubs, [0, 2, 4, 6], [1, 3, 5, 7])

    middle = qc.pull_state()

    fermionic_swap.uncompute()
    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)
    assert not np.allclose(middle, initial)


def test_fermionic_swap_has_desired_action_in_larger_hilbert_space():
    """Test that a fermionic swap has the desired action in a larger Hilbert space."""
    qc = QPU(num_qubits=4)

    qubs = Qubits(4, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    fermionic_swap = FermionicSwapWithReplace()
    fermionic_swap.compute(qubs, [1], [2])

    final_state = qc.pull_state()

    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cz = np.diag([1, 1, 1, -1])

    expected_unitary = swap @ cz
    expected_unitary = np.kron(np.eye(2), np.kron(expected_unitary, np.eye(2)))

    assert np.allclose(final_state, expected_unitary @ initial)


def test_fermionic_swap_does_nothing_when_control_is_off():
    """Test that a fermionic swap does nothing when the control qubit is off."""
    qc = QPU(num_qubits=3)

    one = Qubits(1, "1", qc)
    two = Qubits(1, "2", qc)
    control = Qubits(1, "ctrl", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    control.write(0)
    initial = qc.pull_state()

    fermionic_swap = FermionicSwapWithReplace()
    fermionic_swap.compute(one | two, [0], [1], control)

    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)


def test_nonlocal_fermionic_swap_is_self_inverse():
    """Test that a non-local fermionic swap returns to the initial state when computed twice."""
    qc = QPU(num_qubits=4)

    qubs = Qubits(4, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial_state = qc.pull_state()

    fs = FermionicSwapWithReplace()
    fs.compute(qubs, [0], [3])
    intermediate_state = qc.pull_state()

    # Applying the same non-local swap twice returns to the initial state.
    fs.compute(qubs, [0], [3])
    final_state = qc.pull_state()

    assert not np.allclose(intermediate_state, initial_state)
    assert np.allclose(final_state, initial_state)


def _simulate_expected_basis_vector(initial_state: int, num_qubits: int, a: int, b: int) -> np.ndarray:
    """Simulate the with-replace fermionic swap on a computational-basis vector."""
    bits = [(initial_state >> i) & 1 for i in range(num_qubits)]
    phase = 1

    a, b = (a, b) if a <= b else (b, a)
    for idx in range(b, a, -1):
        if bits[idx] & bits[idx - 1]:
            phase *= -1
        bits[idx], bits[idx - 1] = bits[idx - 1], bits[idx]
    for idx in range(a + 1, b):
        if bits[idx] & bits[idx + 1]:
            phase *= -1
        bits[idx], bits[idx + 1] = bits[idx + 1], bits[idx]

    final_idx = sum((bits[i] << i) for i in range(num_qubits))
    vec = np.zeros(2**num_qubits)
    vec[final_idx] = phase
    return vec


@pytest.mark.parametrize("initial_state, num_qubits", [(0, 4), (8, 4), (5, 4), (22, 5), (38, 6)])
@pytest.mark.parametrize("swap_val_1, swap_val_2", [(0, 3), (1, 3), (0, 2)])
def test_nonlocal_fswap_for_basis_vectors(initial_state, num_qubits, swap_val_1, swap_val_2):
    """Check the exact action of a single non-local fSWAP on computational-basis vectors."""
    qc = QPU(num_qubits=num_qubits)

    qubs = Qubits(num_qubits, "qubs", qc)

    if initial_state != 0:
        qubs.x(initial_state)

    fs = FermionicSwapWithReplace()
    fs.compute(qubs, [swap_val_1], [swap_val_2])

    wf = qc.pull_state()
    expected = _simulate_expected_basis_vector(initial_state, num_qubits, swap_val_1, swap_val_2)

    assert np.allclose(wf, expected)


@pytest.mark.parametrize("num_qubits", [1, 4, 7, 8])
def test_nonlocal_fswap_av_optimized(num_qubits):
    """Test action of optimized non-local fermionic swap has the same action as unoptimized version."""
    # Generate a fSWAP block
    qc = QPU(num_qubits=num_qubits + 2)

    tgt = Qubits(num_qubits + 2, "tgt", qc)

    qc.set_random()
    initial_state = qc.pull_state()

    fswap = FermionicSwapWithReplace()
    fswap.compute(tgt, [tgt[0]], [tgt[-1]])

    wf_ref = qc.pull_state()

    # AV-optimized fSWAP qubrick
    qc = QPU(num_qubits=num_qubits + 2)

    tgt = Qubits(num_qubits + 2, "tgt", qc)

    qc.push_state(initial_state)

    fswap_opt = FermionicSwapWithReplaceAVOpt()
    fswap_opt.compute(tgt, [tgt[0]], [tgt[-1]])

    wf_trial = qc.pull_state()

    assert np.allclose(wf_trial, wf_ref)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_nonlocal_fswap_av_optimized_unitary(num_qubits):
    """Verify decompositions of fSWAP blocks that
    are optimized for AV via unitary matrix filter.
    """
    # Generate a fSWAP block
    unitary = UnitaryMatrixFilter()
    qc = QPU(num_qubits=num_qubits + 2, filters=[unitary])

    tgt = Qubits(num_qubits + 2, "tgt", qc)

    fswap = FermionicSwapWithReplace()
    fswap.compute(tgt, [tgt[0]], [tgt[-1]])

    u_ref = unitary.get()

    # Test decomposition
    unitary = UnitaryMatrixFilter()
    qc = QPU(num_qubits=num_qubits + 2, filters=[unitary])

    tgt = Qubits(num_qubits + 2, "tgt", qc)

    fswap_opt = FermionicSwapWithReplaceAVOpt()
    fswap_opt.compute(tgt, [tgt[0]], [tgt[-1]])

    u_trial = unitary.get()

    assert np.allclose(u_trial, u_ref)
