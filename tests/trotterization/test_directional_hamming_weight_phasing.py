"""Tests for directional Hamming weight phasing."""

import numpy as np
import pytest
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, ComputeHammingWeightNaive
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.arithmetic import NaiveAdd
from psiqdk.workbench.utils.numpy_utils import fidelity, reverse_numpy_op
from scipy.linalg import expm

from psiq_fh.trotterization.catalyst_allocation import allocate_catalyst_registers_PIG, fetch_catalyst_qubits
from psiq_fh.trotterization.directional_hamming_weight_phasing import (
    DirectionalHammingWeightPhasing,
    PowerOfTwoBatchedDirectionalHammingWeightPhasing,
)
from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.trotterization.fermi_hubbard_trotterization import HubbardPlaquetteTrotterizationPIGClosedControl
from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplaceAVOpt
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, TwoModeFFFTViaPPRs, exptXXYY, exptXXYYViaPPR
from psiq_fh.trotterization.interaction import InteractionTrotterStep
from psiq_fh.utils.control_qubit import DirectionalControlQubit
from psiq_fh.utils.fermi_hubbard_hamiltonian import get_fermi_hubbard_hamiltonian_2x2_even_odd


@pytest.mark.parametrize("number_of_target_qubits", [2, 3, 5])
@pytest.mark.parametrize("angle", [11, 435, 164.66692361, -23.43342])
@pytest.mark.parametrize("use_catalyst_state", [True, False])
@pytest.mark.parametrize("rot_is_rz", [True, False])
@pytest.mark.parametrize("preserve_global_phase", [True, False])
def test_directional_hamming_weight_phasing_explicit(
    number_of_target_qubits, angle, use_catalyst_state, rot_is_rz, preserve_global_phase
):
    """Test isolated directional hamming weight phasing reproduces behavior
    of open rotation (-theta) state followed by closed rotation (theta) tower.
    """
    total_number_of_qubits = number_of_target_qubits + int(np.ceil(np.log2(number_of_target_qubits))) + 2

    if use_catalyst_state:
        size_of_catalyst_state = max(1, int(np.ceil(np.log2(number_of_target_qubits))) + 1)
        total_number_of_qubits += 2 * size_of_catalyst_state

    qc = QPU(num_qubits=total_number_of_qubits)
    ctrl = DirectionalControlQubit(1, "ctrl", qc)
    target = Qubits(number_of_target_qubits, "target", qc)

    qc.set_random()
    qc.write(0, ~(ctrl | target).mask())
    init = qc.pull_state()

    # rot tower
    if rot_is_rz:
        target.rz(-1 * angle, ~ctrl)
        target.rz(angle, ctrl)
    else:
        target.phase(-1 * angle, ~ctrl)
        target.phase(angle, ctrl)
    comp_state = qc.pull_state()
    qc.nop(repeat=5)

    # hwp
    qc.push_state(init)

    catalyst_state_reg = None
    if use_catalyst_state:
        cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
        with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg:
            pass

    hwp = DirectionalHammingWeightPhasing(
        angle,
        hamming_weight_qubrick=ComputeHammingWeightNaive(adder=NaiveAdd()),
        use_catalyst_state=use_catalyst_state,
        catalyst_state_reg=catalyst_state_reg,
        rot_is_rz=rot_is_rz,
        preserve_global_phase=preserve_global_phase,
    )

    hwp.compute(target, ctrl=ctrl)
    qc.release_all_rotation_catalyst_qubits()

    final_state = qc.pull_state()

    if preserve_global_phase or not rot_is_rz:
        assert np.allclose(comp_state, final_state)
    else:
        assert np.isclose(fidelity(comp_state, final_state), 1.0)


@pytest.mark.parametrize("ctrl_val", [0, 1])
@pytest.mark.parametrize("number_of_target_qubits", [4, 6, 8])
@pytest.mark.parametrize("preserve_global_phase", [True, False])
def test_batched_directional_hwp(ctrl_val, number_of_target_qubits, preserve_global_phase):
    """Test batched directional hamming weight phasing reproduces behavior of open rotation (-theta) state followed by closed rotation (theta) tower."""
    n_qubits = number_of_target_qubits + 15
    qc = QPU(num_qubits=n_qubits, pre_filters=[">>witness>>"])
    ctrl = Qubits(1, "ctrl", qc)
    tgt = Qubits(number_of_target_qubits, "tgt", qc)
    qc.set_random()
    qc.write(0, ~(tgt).mask())
    ctrl.write(ctrl_val)
    init_state = qc.pull_state()

    # (1) Reference
    angle = 34.6
    tgt.rz(angle, ctrl)
    tgt.rz(-1 * angle, ~ctrl)
    wf_ref = qc.pull_state()

    # (2) Single, larger HWP
    qc = QPU(pre_filters=[">>witness>>"])
    qc.reset(n_qubits)
    ctrl = DirectionalControlQubit(1, "ctrl", qc)
    tgt = Qubits(number_of_target_qubits, "tgt", qc)
    qc.push_state(init_state)
    hamming_weight_qubrick = ComputeHammingWeightNaive(adder=NaiveAdd())
    batched_hamming_weight_qubrick = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick, n_hwp_batches=1, preserve_global_phase=preserve_global_phase
    )
    batched_hamming_weight_qubrick.compute(angle, tgt, ctrl=ctrl)
    qc.release_all_rotation_catalyst_qubits()
    wf_hwp = qc.pull_state()

    # (3) Two, smaller HWPs
    qc = QPU(pre_filters=[">>witness>>"])
    qc.reset(n_qubits)
    ctrl = DirectionalControlQubit(1, "ctrl", qc)
    tgt = Qubits(number_of_target_qubits, "tgt", qc)
    qc.push_state(init_state)
    hamming_weight_qubrick = ComputeHammingWeightNaive(adder=NaiveAdd())
    batched_hamming_weight_qubrick = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick, n_hwp_batches=2, preserve_global_phase=preserve_global_phase
    )
    batched_hamming_weight_qubrick.compute(angle, tgt, ctrl=ctrl)
    qc.release_all_rotation_catalyst_qubits()
    wf_hwp_batched = qc.pull_state()

    # Check output states
    if preserve_global_phase:
        assert np.allclose(wf_ref, wf_hwp)
        assert np.allclose(wf_ref, wf_hwp_batched)
    else:
        assert np.isclose(fidelity(wf_ref, wf_hwp), 1)
        assert np.isclose(fidelity(wf_ref, wf_hwp_batched), 1)


@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize(
    "potential_coefficient, kinetic_coefficient, total_evolution_time, number_of_trotter_steps",
    [
        (8, 1, 0.1, 10),
    ],
)
@pytest.mark.parametrize("n_hwp_batches", [1, 2])
@pytest.mark.parametrize("preserve_global_phase", [True, False])
def test_closed_control_trotter_qbk_with_no_ctrl(
    use_ppr,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    number_of_trotter_steps,
    n_hwp_batches,
    preserve_global_phase,
):
    """Test closed controlled evolution qubrick does not work on no control."""
    x_dim, y_dim = 2, 2
    number_spin_sites = 2 * x_dim * y_dim
    hw_qubits = 15
    total_num_qubits = number_spin_sites + hw_qubits + 1
    particle_hole_symmetry = True

    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=False)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=False)

    qc = QPU(num_qubits=total_num_qubits)
    psi_reg = Qubits(number_spin_sites, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    qc.write(0, ~(psi_reg).mask())

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=None,  # default
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )

    # call directional hwp
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick, n_hwp_batches=n_hwp_batches, rot_is_rz=True, preserve_global_phase=preserve_global_phase
    )

    if use_ppr:
        exptXXYY_qbk = exptXXYYViaPPR(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    else:
        exptXXYY_qbk = exptXXYY(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick, two_mode_ffft_qubrick, exptXXYY_qubrick=exptXXYY_qbk
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick, two_mode_ffft_qubrick, exptXXYY_qubrick=exptXXYY_qbk
    )

    hubbard_time_evolution = HubbardPlaquetteTrotterizationPIGClosedControl(
        interaction_trotter_step, pink_plaquette_trotter_step, gold_plaquette_trotter_step
    )

    allocate_catalyst_registers_PIG(
        x_dim,
        y_dim,
        potential_coefficient,
        kinetic_coefficient,
        total_evolution_time,
        number_of_trotter_steps,
        n_hwp_batches,
        qc,
    )
    catalyst_reg = fetch_catalyst_qubits(qc)

    with pytest.raises(ValueError):
        hubbard_time_evolution.compute(
            psi_reg,
            catalyst_reg,
            fh_data,
            ctrl=0,
        )
