"""Tests for the XXYY evolution in the Fermi-Hubbard Trotterization circuit in arxiv:2012.09238."""

import numpy as np
import pytest
from psiqdk.algorithms.utils import PauliMask, PauliSum, pauli_sum_to_numpy
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.utility_filters import UnitaryMatrixFilter
from scipy.linalg import expm

from psiq_fh.trotterization.hopping import exptXXYY, exptXXYYViaPPR


@pytest.mark.parametrize("tau", [-50, -1.783, 0, 75])
@pytest.mark.parametrize("use_ppr", [True, False])
def test_action_of_xxyy_evolution_using_gates(tau, use_ppr):
    """Test the action of the XXYY evolution using gates, with and without PPRs."""
    xx = pauli_sum_to_numpy(PauliSum([1, PauliMask.from_pauli_string("X0 X1")]))
    yy = pauli_sum_to_numpy(PauliSum([1, PauliMask.from_pauli_string("Y0 Y1")]))

    time_evolution_of_xxyy = expm(1j * tau * (xx)) @ expm(1j * tau * (yy))

    qc = QPU(num_qubits=2)

    qubs = Qubits(2, "qubs", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    if use_ppr:
        expXXYY = exptXXYYViaPPR()
    else:
        expXXYY = exptXXYY()

    rz_rotation_angle = -1 * 2 * tau * 180 / np.pi  # See eqn E13 of https://arxiv.org/abs/2012.09238

    expXXYY.compute(qubs[0], qubs[1], rz_rotation_angle, ctrl=0)

    final_state = qc.pull_state()

    assert np.allclose(final_state, time_evolution_of_xxyy @ initial)


@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize("tau", [-50, -1.783, 0, 75])
def test_xxyy_does_nothing_when_control_is_zero(tau, use_ppr):
    """Test that the XXYY evolution does nothing when the control is zero."""
    qc = QPU(num_qubits=3)

    control = Qubits(1, "ctrl", qc)
    qubs = Qubits(2, "qubs", qc)
    qc.set_param("random_seed", 42)
    qc.set_random()
    control.write(0)
    initial = qc.pull_state()

    if use_ppr:
        expXXYY = exptXXYYViaPPR()
    else:
        expXXYY = exptXXYY()

    rz_rotation_angle = -1 * 2 * tau * 180 / np.pi  # See eqn E13 of https://arxiv.org/abs/2012.09238

    expXXYY.compute(qubs[0], qubs[1], rz_rotation_angle, ctrl=control)
    final_state = qc.pull_state()

    assert np.allclose(final_state, initial)


@pytest.mark.parametrize("n_parallel", [1, 3])
def test_unitary_of_exptXXYY_with_and_without_PPRs(n_parallel):
    """Test application of parallel exptXXYY term(s) where
    n_parallel is the number of parallel terms.
    """
    angle = 85.111
    n_qubits = 2 * n_parallel

    # Gate-based
    UNITARY = UnitaryMatrixFilter()
    qc = QPU(num_qubits=n_qubits, pre_filters=[UNITARY])
    UNITARY.qc = qc
    UNITARY.clear()

    tgts = Qubits(n_qubits, "tgts", qc)

    site_2_qubits = tgts[::2]
    site_3_qubits = tgts[1::2]

    qbk = exptXXYY()
    qbk.compute(site_2_qubits, site_3_qubits, angle)

    gate_u = UNITARY.get()

    # PPR-based
    UNITARY = UnitaryMatrixFilter()
    qc = QPU(num_qubits=n_qubits, pre_filters=[UNITARY])
    UNITARY.qc = qc
    UNITARY.clear()

    tgts = Qubits(n_qubits, "tgts", qc)

    site_2_qubits = tgts[::2]

    site_3_qubits = tgts[1::2]

    qbk = exptXXYYViaPPR()
    qbk.compute(site_2_qubits, site_3_qubits, angle)

    ppr_u = UNITARY.get()

    assert np.allclose(gate_u, ppr_u)
