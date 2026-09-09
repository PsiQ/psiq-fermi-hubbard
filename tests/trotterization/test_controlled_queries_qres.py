"""Tests for QREs of closed-controlled and directionally controlled Trotterized time evolution operators."""

from unittest.mock import patch

import numpy as np
import pytest
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees
from psiqdk.workbench import QPU, Qubits

from psiq_fh.trotterization.catalyst_allocation import allocate_catalyst_registers_PIG, fetch_catalyst_qubits
from psiq_fh.trotterization.directional_hamming_weight_phasing import (
    PowerOfTwoBatchedDirectionalHammingWeightPhasing,
)
from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.trotterization.fermi_hubbard_trotterization import (
    HubbardPlaquetteTrotterizationPIG,
    HubbardPlaquetteTrotterizationPIGClosedControl,
)
from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplaceAVOpt
from psiq_fh.trotterization.fswapping.fswap_network import PinkLocalizedFermionicSwapNetworkWithReplace
from psiq_fh.utils.jw_ordering_utils import generate_high_low_enum
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, TwoModeFFFTViaPPRs, exptXXYY, exptXXYYViaPPR
from psiq_fh.trotterization.interaction import InteractionTrotterStep
from psiq_fh.utils.control_qubit import ControlQubit, DirectionalControlQubit


@pytest.mark.parametrize("x_dim", [4, 6])
@pytest.mark.parametrize("potential_coefficient", [8])
@pytest.mark.parametrize("kinetic_coefficient", [1])
@pytest.mark.parametrize("total_evolution_time", [0.32])
@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize("number_of_trotter_steps", [4, 7])  # even and odd
@pytest.mark.parametrize("n_hwp_batches", [1, 4])
def test_closed_control_trotterized_evolution(
    x_dim,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    use_ppr,
    number_of_trotter_steps,
    n_hwp_batches,
):
    """Test circuit structure of closed-controlled Trotterized evolution."""
    y_dim = x_dim

    number_spin_sites = 2 * x_dim * y_dim
    hw_qubits = 1 * number_spin_sites
    total_num_qubits = number_spin_sites + hw_qubits + 1
    particle_hole_symmetry = True

    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=False)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=False)

    qc = QPU(filters=[">>witness>>", ">>buffer>>"])
    qc.reset(total_num_qubits)
    psi_reg = Qubits(number_spin_sites, "psi", qc)
    ctrl = Qubits(1, "ctrl", qc)

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=generate_high_low_enum(x_dim, L_y=y_dim, pink_happy=True),
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )

    # call directional hwp
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick,
        n_hwp_batches=n_hwp_batches,
        rot_is_rz=True,
        preserve_global_phase=False,
        turn_on_cnots=False,  # Disable CNOTs on catalysts called for each HWP
    )

    if use_ppr:
        exptXXYY_qbk = exptXXYYViaPPR(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    else:
        exptXXYY_qbk = exptXXYY(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    localize_plaquettes_qubrick = PinkLocalizedFermionicSwapNetworkWithReplace(fermionic_swap_qubrick)  # Pink-happy
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )

    # Pre-allocate catalysts
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
    cat_qubits = fetch_catalyst_qubits(qc)

    # Keep track of term names, ctrl type, and evolution times
    records = []

    def record_and_call(name, original_compute):
        """Keep track of Trotter term names, control type, and evolution times.

        Args:
            name (str): name for Trotter term, e.g. "pink", "gold", "int"
            original_compute (Callable): compute method for each type of Trotter term
        """

        def _side_effect(target_reg, data, **kwargs):
            records.append(
                {
                    "name": name,
                    "ctrl": kwargs.get("ctrl"),
                    "evolution_time": data.evolution_time,
                }
            )
            return original_compute(target_reg, data, **kwargs)

        return _side_effect

    # Track information on ctrl and evolution times
    orig_pink_compute = pink_plaquette_trotter_step.compute
    orig_int_compute = interaction_trotter_step.compute
    orig_gold_compute = gold_plaquette_trotter_step.compute

    with (
        patch.object(pink_plaquette_trotter_step, "compute") as pink_mock,
        patch.object(interaction_trotter_step, "compute") as int_mock,
        patch.object(gold_plaquette_trotter_step, "compute") as gold_mock,
    ):
        pink_mock.side_effect = record_and_call("pink", orig_pink_compute)
        int_mock.side_effect = record_and_call("int", orig_int_compute)
        gold_mock.side_effect = record_and_call("gold", orig_gold_compute)

        # Call qubrick
        hubbard_time_evolution = HubbardPlaquetteTrotterizationPIGClosedControl(
            interaction_trotter_step,
            pink_plaquette_trotter_step,
            gold_plaquette_trotter_step,
        )
        hubbard_time_evolution.compute(psi_reg, cat_qubits, fh_data, ctrl=ctrl)

    # Check number of catalysts
    assert cat_qubits.num_qubits == 2 * int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 3

    ctrls = [rec["ctrl"] for rec in records]
    times = [rec["evolution_time"] for rec in records]
    names = [rec["name"] for rec in records]

    ### Check control structure ###

    # Check that first 2r steps are uncontrolled (i.e. ctrl = 0), where r = number of Trotter steps
    assert all(x == 0 for x in ctrls[: 2 * number_of_trotter_steps])

    # Check that we should only have n_hwp_batches number of controlled phase gate from the
    # center Trotter term that is closed controlled (i.e. from controlling payload rotation(s))
    assert qc.witness.filter("qc.phase", target=1, condition=1).count() == n_hwp_batches

    # Check that middle step is standard closed controlled (i.e. ctrl is type ControlQubit)
    assert isinstance(ctrls[2 * number_of_trotter_steps], ControlQubit)

    # Check that last 2r steps are directionally controlled (i.e. ctrl is type DirectionalControlQubit)
    assert all(isinstance(x, DirectionalControlQubit) for x in ctrls[2 * number_of_trotter_steps + 1 :])

    ### Check Trotter sequence
    pattern = ["pink", "int", "gold", "int"]  # in this order for `PIG'
    assert all(name == pattern[i % len(pattern)] for i, name in enumerate(names))
    assert len(names) == 4 * number_of_trotter_steps + 1

    ### Check relative evolution times (not rotation angles so not including u, t)
    int_times = np.array(times[1:-1:2])
    assert np.all(int_times == int_times.flat[0])  # check interaction (int) term times are all equal

    intermed_pink_gold_times = np.array(times[2:-2:2])
    assert np.all(
        intermed_pink_gold_times == intermed_pink_gold_times.flat[0]
    )  # check intermediate pink and all gold term times are equal
    assert np.isclose(
        intermed_pink_gold_times.flat[0], 2 * times[0]
    )  # merged pink / gold times (first and last) twice that of unmerged pink time
    assert np.isclose(intermed_pink_gold_times.flat[-1], 2 * times[-1])


@pytest.mark.parametrize("x_dim", [4, 6])
@pytest.mark.parametrize("potential_coefficient", [8])
@pytest.mark.parametrize("kinetic_coefficient", [1])
@pytest.mark.parametrize("total_evolution_time", [0.32])
@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize("number_of_trotter_steps", [4, 7])  # even and odd
@pytest.mark.parametrize("n_hwp_batches", [1, 4])
@pytest.mark.parametrize("power", [1, 2, 5])
def test_directionally_controlled_trotterized_evolution(
    x_dim,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    use_ppr,
    number_of_trotter_steps,
    n_hwp_batches,
    power,
):
    """Test circuit structure of directionally-controlled Trotterized evolution."""
    y_dim = x_dim
    number_spin_sites = 2 * x_dim * y_dim
    hw_qubits = 2 * number_spin_sites
    total_num_qubits = number_spin_sites + hw_qubits + 1
    particle_hole_symmetry = True

    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=False)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=False)

    qc = QPU(filters=[">>witness>>", ">>buffer>>"])
    qc.reset(total_num_qubits)
    psi_reg = Qubits(number_spin_sites, "psi", qc)

    ctrl = DirectionalControlQubit(1, "ctrl", qc, defer_rotations=True)

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=generate_high_low_enum(x_dim, L_y=y_dim, pink_happy=True),
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )

    # call directional hwp
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick,
        n_hwp_batches=n_hwp_batches,
        rot_is_rz=True,
        preserve_global_phase=False,
        turn_on_cnots=False,  # Disable CNOTs on catalysts called for each HWP
    )

    if use_ppr:
        exptXXYY_qbk = exptXXYYViaPPR(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    else:
        exptXXYY_qbk = exptXXYY(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    localize_plaquettes_qubrick = PinkLocalizedFermionicSwapNetworkWithReplace(fermionic_swap_qubrick)  # Pink-happy
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )

    # Pre-allocate catalysts
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
    cat_qubits = fetch_catalyst_qubits(qc)

    # Keep track of term names, ctrl type, and evolution times
    records = []

    def record_and_call(name, original_compute):
        def _side_effect(target_reg, data, **kwargs):
            records.append(
                {
                    "name": name,
                    "ctrl": kwargs.get("ctrl"),
                    "evolution_time": data.evolution_time,
                }
            )
            return original_compute(target_reg, data, **kwargs)

        return _side_effect

    # Track information on ctrl and evolution times
    orig_pink_compute = pink_plaquette_trotter_step.compute
    orig_int_compute = interaction_trotter_step.compute
    orig_gold_compute = gold_plaquette_trotter_step.compute

    with (
        patch.object(pink_plaquette_trotter_step, "compute") as pink_mock,
        patch.object(interaction_trotter_step, "compute") as int_mock,
        patch.object(gold_plaquette_trotter_step, "compute") as gold_mock,
    ):
        pink_mock.side_effect = record_and_call("pink", orig_pink_compute)
        int_mock.side_effect = record_and_call("int", orig_int_compute)
        gold_mock.side_effect = record_and_call("gold", orig_gold_compute)

        # Call qubrick
        hubbard_time_evolution = HubbardPlaquetteTrotterizationPIG(
            interaction_trotter_step,
            pink_plaquette_trotter_step,
            gold_plaquette_trotter_step,
        )
        hubbard_time_evolution.compute(psi_reg, fh_data, cat_qubits, power=power, ctrl=ctrl)

    ctrls = [rec["ctrl"] for rec in records]
    times = [rec["evolution_time"] for rec in records]
    names = [rec["name"] for rec in records]

    # Check that every term is directionally controlled
    assert all(isinstance(x, DirectionalControlQubit) for x in ctrls)

    assert cat_qubits.num_qubits == 2 * int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 3

    # Check Trotter sequence
    pattern = ["pink", "int", "gold", "int"]  # in this order for `PIG'
    assert all(name == pattern[i % len(pattern)] for i, name in enumerate(names))

    # Check number of calls to P, I, and G terms!
    assert names.count("pink") == power * number_of_trotter_steps + 1
    assert names.count("int") == power * 2 * number_of_trotter_steps
    assert names.count("gold") == power * number_of_trotter_steps

    # Should only have 2 CNOT fanouts acting on catalyst qubits!
    assert qc.witness.filter("qc.x", target=cat_qubits.num_qubits, condition=1).count() == 2

    ### Check relative evolution times (not rotation angles so not including u, t)
    int_times = np.array(times[1:-1:2])
    assert np.all(int_times == int_times.flat[0])  # check interaction (int) term times are all equal

    pink_gold_times = np.array(times[2:-2:2])
    assert np.all(
        pink_gold_times == pink_gold_times.flat[0]
    )  # check intermediate pink and all gold term times are equal
    assert np.isclose(
        pink_gold_times.flat[0], 2 * times[0]
    )  # merged pink / gold times (first and last) twice that of unmerged pink time
    assert np.isclose(pink_gold_times.flat[-1], 2 * times[-1])

    # No controlled phases (i.e. no closed controlled HWP)
    assert qc.witness.filter("qc.phase", target=1, condition=1).count() == 0


def _setup_directionally_controlled_trotterized_evolution(
    x_dim,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    use_ppr,
    number_of_trotter_steps,
    n_hwp_batches,
    power,
    use_jump_back=False,
):
    """Set up directionally controlled Trotterized time evolution and return metrics."""
    y_dim = x_dim
    number_spin_sites = 2 * x_dim * y_dim
    hw_qubits = 2 * number_spin_sites
    total_num_qubits = number_spin_sites + hw_qubits + 1
    particle_hole_symmetry = True

    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=False)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=False)

    qc = QPU(filters=[">>witness>>", ">>buffer>>"])
    qc.reset(total_num_qubits)
    psi_reg = Qubits(number_spin_sites, "psi", qc)

    ctrl = DirectionalControlQubit(1, "ctrl", qc, defer_rotations=True)

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=generate_high_low_enum(x_dim, L_y=y_dim, pink_happy=True),
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )

    # call directional hwp
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick,
        n_hwp_batches=n_hwp_batches,
        rot_is_rz=True,
        preserve_global_phase=False,
        turn_on_cnots=False,  # Disable CNOTs on catalysts called for each HWP
    )

    if use_ppr:
        exptXXYY_qbk = exptXXYYViaPPR(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    else:
        exptXXYY_qbk = exptXXYY(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    localize_plaquettes_qubrick = PinkLocalizedFermionicSwapNetworkWithReplace(fermionic_swap_qubrick)  # Pink-happy
    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )

    # Pre-allocate catalysts
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
    cat_qubits = fetch_catalyst_qubits(qc)

    # Call qubrick
    hubbard_time_evolution = HubbardPlaquetteTrotterizationPIG(
        interaction_trotter_step,
        pink_plaquette_trotter_step,
        gold_plaquette_trotter_step,
    )
    hubbard_time_evolution.compute(psi_reg, fh_data, cat_qubits, power=power, ctrl=ctrl, use_jump_back=use_jump_back)
    return qc


@pytest.mark.parametrize("x_dim", [4])
@pytest.mark.parametrize("potential_coefficient", [8])
@pytest.mark.parametrize("kinetic_coefficient", [1])
@pytest.mark.parametrize("total_evolution_time", [0.32])
@pytest.mark.parametrize("use_ppr", [True, False])
@pytest.mark.parametrize("number_of_trotter_steps", [3, 4])  # even and odd
@pytest.mark.parametrize("n_hwp_batches", [1, 4])
@pytest.mark.parametrize("power", [2, 3])
def test_jumpback_for_directionally_controlled_trotterized_evolution(
    x_dim,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    use_ppr,
    number_of_trotter_steps,
    n_hwp_batches,
    power,
):
    """Compare numeric QREs between naively counting metrics vs. using jump back for NQRE speedups."""
    qc_normal = _setup_directionally_controlled_trotterized_evolution(
        x_dim,
        potential_coefficient,
        kinetic_coefficient,
        total_evolution_time,
        use_ppr,
        number_of_trotter_steps,
        n_hwp_batches,
        power,
    )
    qc_jump = _setup_directionally_controlled_trotterized_evolution(
        x_dim,
        potential_coefficient,
        kinetic_coefficient,
        total_evolution_time,
        use_ppr,
        number_of_trotter_steps,
        n_hwp_batches,
        power,
        use_jump_back=True,
    )
    metrics_normal = qc_normal.metrics()
    metrics_jump = qc_jump.metrics()

    assert metrics_normal == metrics_jump


@pytest.mark.parametrize("x_dim", [6])
@pytest.mark.parametrize("potential_coefficient", [8])
@pytest.mark.parametrize("kinetic_coefficient", [1])
@pytest.mark.parametrize("total_evolution_time", [0.32])
@pytest.mark.parametrize("use_ppr", [True])
@pytest.mark.parametrize("number_of_trotter_steps", [3, 4])  # even and odd
@pytest.mark.parametrize("n_hwp_batches", [1, 4])
@pytest.mark.parametrize("n_phase_qubits", [2, 3])
def test_jumpback_qres_for_directional_qpe(
    x_dim,
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    use_ppr,
    number_of_trotter_steps,
    n_hwp_batches,
    n_phase_qubits,
):
    """Test larger NQREs of the entire QPE control structure using jump back."""
    y_dim = x_dim
    number_spin_sites = 2 * x_dim * y_dim
    hw_qubits = 2 * number_spin_sites
    total_num_qubits = number_spin_sites + hw_qubits + 1
    particle_hole_symmetry = True

    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=False)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=False)

    qc = QPU(filters=[">>witness>>", ">>buffer>>"])
    qc.reset(total_num_qubits)

    phase_reg = Qubits(n_phase_qubits, "phase_reg", qc)

    psi_reg = Qubits(number_spin_sites, "psi", qc)

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=generate_high_low_enum(x_dim, L_y=y_dim, pink_happy=True),
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )

    # call directional hwp
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hamming_weight_qubrick,
        n_hwp_batches=n_hwp_batches,
        rot_is_rz=True,
        preserve_global_phase=False,
        turn_on_cnots=False,  # Disable CNOTs on catalysts called for each HWP
    )

    if use_ppr:
        exptXXYY_qbk = exptXXYYViaPPR(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    else:
        exptXXYY_qbk = exptXXYY(batched_hamming_weight_phasing_qubrick=hwp_qbk)

    localize_plaquettes_qubrick = PinkLocalizedFermionicSwapNetworkWithReplace(fermionic_swap_qubrick)

    interaction_trotter_step = InteractionTrotterStep(batched_hamming_weight_phasing_qubrick=hwp_qbk)
    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick,
        two_mode_ffft_qubrick,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )

    # Pre-allocate catalysts
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
    cat_qubits = fetch_catalyst_qubits(qc)

    U_closed_control = HubbardPlaquetteTrotterizationPIGClosedControl(
        interaction_trotter_step,
        pink_plaquette_trotter_step,
        gold_plaquette_trotter_step,
    )

    U_powered_directional = HubbardPlaquetteTrotterizationPIG(
        interaction_trotter_step,
        pink_plaquette_trotter_step,
        gold_plaquette_trotter_step,
    )

    # Call qubricks
    ctrl = DirectionalControlQubit(phase_reg[0], defer_rotations=True)
    U_closed_control.compute(psi_reg, catalyst_reg=cat_qubits, data=fh_data, ctrl=ctrl)

    for j, qbit in enumerate(phase_reg[1:]):
        power = 2**j
        qbit = DirectionalControlQubit(qbit, defer_rotations=True)
        U_powered_directional.compute(
            psi_reg, fh_data, catalyst_reg=cat_qubits, power=power, ctrl=qbit, use_jump_back=True
        )

    assert cat_qubits.num_qubits == 2 * int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 3

    # Check total numbers of P, I, G calls (comparing with analytically computed values)
    pink_tot = number_of_trotter_steps * 2 ** (n_phase_qubits - 1) + n_phase_qubits
    int_tot = number_of_trotter_steps * 2 ** (n_phase_qubits)
    gold_tot = number_of_trotter_steps * 2 ** (n_phase_qubits - 1)

    witness_int_count = qc.witness.filter(label="InteractionTrotterStep", name="qc.qbk_compute_start").count()
    assert witness_int_count == int_tot
    witness_pink_gold_count = qc.witness.filter(label="PlaquetteTrotterStep", name="qc.qbk_compute_start").count()
    assert witness_pink_gold_count == pink_tot + gold_tot

    # Check that number of CNOT fanouts on catalysts only happens 2 * n_phase_qubits number of times
    assert qc.witness.filter("qc.x", target=cat_qubits.num_qubits, condition=1).count() == 2 * n_phase_qubits

    metrics = qc.metrics()
    chad_count = 2 * x_dim * y_dim  # per plaq call
    assert metrics["t_count"] == 2 * chad_count * witness_pink_gold_count
