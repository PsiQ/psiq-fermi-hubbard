"""Tests for control qubit types."""

import numpy as np
import pytest
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.utils.numpy_utils import fidelity

from psiq_fh.utils.control_qubit import ControlQubit, DirectionalControlQubit


def test_control_qubit_types_can_only_be_one_qubit():
    """Test that control qubit types can only be one qubit."""
    qc = QPU(num_qubits=2)
    qbts = Qubits(2, "ctrls", qc)

    with pytest.raises(RuntimeError):
        ControlQubit(qbts)

    with pytest.raises(RuntimeError):
        DirectionalControlQubit(qbts)


def test_from_control_method_doesnt_mutate():
    """Test that directional control qubit converted from control qubit
    does not mutate the original control qubit.
    """
    qc = QPU(num_qubits=1)
    qbts = Qubits(1, "ctrl", qc)

    a = ControlQubit(qbts)
    b = DirectionalControlQubit.from_control(a)

    assert isinstance(b, DirectionalControlQubit)
    assert not isinstance(a, DirectionalControlQubit)
    assert isinstance(a, ControlQubit)


@pytest.mark.parametrize("op_type", ["rz", "phase", "reflect_z"])
@pytest.mark.parametrize("is_directional", [True, False])
def test_defer_rotation(op_type, is_directional):
    """Test deferral of rotations for both control qubit types."""
    rng = np.random.default_rng(42)
    rot_angles = rng.uniform(size=4)

    qc = QPU(num_qubits=4)

    ctrl = Qubits(1, "ctrl", qc)
    tgts = Qubits(2, "tgts", qc)

    if is_directional:
        ctrl = DirectionalControlQubit(ctrl, defer_rotations=True)
    else:
        ctrl = ControlQubit(ctrl, defer_rotations=True)

    if op_type == "phase":
        for rot_angle in rot_angles:
            ctrl.phase(rot_angle)
    elif op_type == "rz":
        for rot_angle in rot_angles:
            ctrl.rz(rot_angle)
    elif op_type == "reflect_z":
        for rot_angle in rot_angles:
            ctrl.reflect(rot_angle)

    tgts.x(ctrl)
    tgts[0].phase(35)  # try to add another rotation but not acting on control
    ctrl.resolve_rotations()

    if op_type == "phase":
        filtered_op = qc.witness.filter(
            name=lambda x: x == "qc.phase", theta=lambda x: np.isclose(x, np.sum(rot_angles))
        )
    elif op_type == "rz":
        filtered_op = qc.witness.filter(name=lambda x: x == "qc.rz", theta=lambda x: np.isclose(x, np.sum(rot_angles)))
    elif op_type == "reflect_z":
        # On a single qubit, reflect on z-axis is just phase
        filtered_op = qc.witness.filter(
            name=lambda x: x == "qc.phase", theta=lambda x: np.isclose(x, np.sum(rot_angles))
        )

    # Check that merged into one rotation op
    assert filtered_op.count() == 1

    # Check the merged angle
    merged_rotation_angle = list(filtered_op.witness_dict.keys())[0].args["theta"]
    assert np.isclose(merged_rotation_angle, np.sum(rot_angles))


@pytest.mark.parametrize("is_directional", [True, False])
def test_defer_rotation_merging_phase_and_rz(is_directional):
    """Test combining phase and RZ merged rotations into one RZ rotation."""
    rng = np.random.default_rng(42)
    phase_rot_angles = rng.uniform(size=4)
    rz_rot_angles = rng.uniform(size=4)

    # (1) Merge RZ and phase
    qc = QPU(num_qubits=3)

    ctrl = Qubits(1, "ctrl", qc)
    tgts = Qubits(2, "tgts", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    init_state = qc.pull_state()

    if is_directional:
        ctrl = DirectionalControlQubit(ctrl, defer_rotations=True)
    else:
        ctrl = ControlQubit(ctrl, defer_rotations=True)

    for phase_angle, rz_angle in zip(phase_rot_angles, rz_rot_angles):
        ctrl.phase(phase_angle)
        ctrl.rz(rz_angle)

    tgts.x(ctrl)
    ctrl.resolve_rotations(merge_phase_and_rz=True)

    wf_merging = qc.pull_state()

    qc.draw()

    # (2) Don't merge
    qc = QPU(num_qubits=3)

    ctrl = Qubits(1, "ctrl", qc)
    tgts = Qubits(2, "tgts", qc)

    qc.push_state(init_state)

    if is_directional:
        ctrl = DirectionalControlQubit(ctrl, defer_rotations=True)
    else:
        ctrl = ControlQubit(ctrl, defer_rotations=True)

    for phase_angle, rz_angle in zip(phase_rot_angles, rz_rot_angles):
        ctrl.phase(phase_angle)
        ctrl.rz(rz_angle)

    tgts.x(ctrl)
    ctrl.resolve_rotations(merge_phase_and_rz=False)

    wf_no_merging = qc.pull_state()

    qc.draw()

    # Equal up to global phase
    assert np.isclose(fidelity(wf_merging, wf_no_merging), 1)


@pytest.mark.parametrize("op_type", ["rz", "phase", "reflect_z"])
@pytest.mark.parametrize("is_directional", [True, False])
def test_skip_rotation(op_type, is_directional):
    """Test that no rotations are emitted."""
    rng = np.random.default_rng(42)
    rot_angles = rng.uniform(size=4)

    qc = QPU(num_qubits=4)

    ctrl = Qubits(1, "ctrl", qc)
    tgts = Qubits(2, "tgts", qc)

    if is_directional:
        ctrl = DirectionalControlQubit(ctrl, skip_rotations=True)
    else:
        ctrl = ControlQubit(ctrl, skip_rotations=True)

    if op_type == "phase":
        for rot_angle in rot_angles:
            ctrl.phase(rot_angle)
    elif op_type == "rz":
        for rot_angle in rot_angles:
            ctrl.rz(rot_angle)
    elif op_type == "reflect_z":
        for rot_angle in rot_angles:
            ctrl.reflect(rot_angle)

    tgts.x(ctrl)
    tgts[0].phase(35)  # try to add another rotation but not acting on control
    ctrl.resolve_rotations()

    if op_type == "phase":
        filtered_op = qc.witness.filter(
            name=lambda x: x == "qc.phase", theta=lambda x: np.isclose(x, np.sum(rot_angles))
        )
    elif op_type == "rz":
        filtered_op = qc.witness.filter(name=lambda x: x == "qc.rz", theta=lambda x: np.isclose(x, np.sum(rot_angles)))
    elif op_type == "reflect_z":
        # On a single qubit, reflect on z-axis is just phase
        filtered_op = qc.witness.filter(
            name=lambda x: x == "qc.phase", theta=lambda x: np.isclose(x, np.sum(rot_angles))
        )

    # Check that merged into one rotation op
    assert filtered_op.count() == 0
