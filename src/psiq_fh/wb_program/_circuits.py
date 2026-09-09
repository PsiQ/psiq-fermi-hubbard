from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from psiqdk.algorithms import (
    ComputeHammingWeightGroupOfThrees,
    PowerOfTwoBatchedHammingWeightPhasing,
)
from psiqdk.workbench import QPU, Qubits

from psiq_fh.trotterization.catalyst_allocation import (
    allocate_catalyst_registers_IPG,
    allocate_catalyst_registers_PIG,
    fetch_catalyst_qubits,
)
from psiq_fh.trotterization.directional_hamming_weight_phasing import (
    PowerOfTwoBatchedDirectionalHammingWeightPhasing,
)
from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.trotterization.fermi_hubbard_trotterization import (
    HubbardPlaquetteTrotterizationIPG,
    HubbardPlaquetteTrotterizationPIG,
    HubbardPlaquetteTrotterizationPIGClosedControl,
)
from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt
from psiq_fh.trotterization.fswapping.fswap_network import (
    NaivefSWAPNetworkWithReplace,
    PinkLocalizedFermionicSwapNetworkWithReplace,
)
from psiq_fh.trotterization.hopping import (
    PlaquetteTrotterStep,
    TwoModeFFFTViaControlledHad,
    TwoModeFFFTViaPPRs,
    exptXXYY,
    exptXXYYViaPPR,
)
from psiq_fh.trotterization.interaction import InteractionTrotterStep
from psiq_fh.utils.control_qubit import ControlQubit, DirectionalControlQubit


@dataclass
class CircuitData:
    """Data class for circuit data.

    Attributes:
        qpe_energy_error: Energy error from QPE.
        trotter_energy_error: Energy error from Trotterization.
        rotation_energy_error: Energy error from (non-catalyst) rotations.
        cat_energy_error: Energy error from catalyst rotations.
        evolution_time: Total evolution time.
        no_trotter_steps: Number of Trotter steps.
        rot_bop: Precision bits for (non-catalyst) rotations.
        no_queries: Number of queries in QPE.
        cat_bop: Precision bits for catalyst rotations.
        toffoli_estimate: Estimate of the number of Toffoli gates (assumes no term merging).
        no_batches: Number of Hamming weight phasing batches.
        ordering: Trotter term ordering, either "PIG" or "IPG".
        x: Array of error allocation parameters.
    """

    qpe_energy_error: float
    trotter_energy_error: float
    rotation_energy_error: float
    cat_energy_error: float
    evolution_time: float
    no_trotter_steps: int
    rot_bop: int
    no_queries: int
    cat_bop: int
    toffoli_estimate: float
    no_batches: int
    ordering: Literal["PIG", "IPG"]
    x: Iterable[float]

    def __getitem__(self, key: str) -> int | float | str:
        return getattr(self, key)


def build_baseline_qubricks(num_batches: int = 1) -> HubbardPlaquetteTrotterizationIPG:
    """Instantiate the baseline qubricks.

    Args:
        num_batches (int, optional): Number of Hamming weight phasing batches. Defaults to 1.

    Returns:
        HubbardPlaquetteTrotterizationIPG: Trotterized time evolution operator using IPG term ordering.
    """
    hamming_weight_qbk = ComputeHammingWeightGroupOfThrees()
    hw_phasing_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qbk, n_hwp_batches=num_batches)

    interaction_trotter_step = InteractionTrotterStep(hw_phasing_qbk)
    exptXXYY_qbk = exptXXYY(hw_phasing_qbk)
    two_mode_ffft_qbk = TwoModeFFFTViaControlledHad()
    fswap_block_qbk = FermionicSwapWithReplace()
    fswap_network_qbk = NaivefSWAPNetworkWithReplace(fswap_block_qbk)

    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick=fswap_block_qbk,
        two_mode_ffft_qubrick=two_mode_ffft_qbk,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=fswap_network_qbk,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick=fswap_block_qbk,
        two_mode_ffft_qubrick=two_mode_ffft_qbk,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=fswap_network_qbk,
    )

    return HubbardPlaquetteTrotterizationIPG(
        interaction_trotter_step,
        pink_plaquette_trotter_step,
        gold_plaquette_trotter_step,
    )


def build_improved_qubricks(
    num_batches: int = 1, use_black_box: bool = True
) -> tuple[HubbardPlaquetteTrotterizationPIGClosedControl, HubbardPlaquetteTrotterizationIPG]:
    """Instantiate the improved qubricks.

    Args:
        num_batches (int, optional): Number of Hamming weight phasing batches. Defaults to 1.
        use_black_box (bool, optional): Whether to use black box qubricks for ZX-optimized AV counts. Defaults to True.

    Returns:
        tuple[HubbardPlaquetteTrotterizationPIGClosedControl, HubbardPlaquetteTrotterizationIPG]:
            Trotterized time evolution operators using PIG term ordering with closed control and directional control.
    """
    hw_qbk = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
        hw_qbk,
        n_hwp_batches=num_batches,
        rot_is_rz=True,
        preserve_global_phase=False,
        turn_on_cnots=False,  # Disable CNOTs on catalysts called for each HWP
        use_black_box=use_black_box,  # blackbox for adders
    )

    interaction_trotter_step = InteractionTrotterStep(hwp_qbk)
    exptXXYY_qbk = exptXXYYViaPPR(hwp_qbk)
    two_mode_ffft_qbk = TwoModeFFFTViaPPRs(use_black_box=use_black_box)
    fswap_block_qbk = FermionicSwapWithReplaceAVOpt(
        use_black_box=use_black_box
    )  # bb won't show on circuit diagram (qc.draw)
    localize_plaquettes_qubrick = PinkLocalizedFermionicSwapNetworkWithReplace(fswap_block_qbk)

    pink_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick=fswap_block_qbk,
        two_mode_ffft_qubrick=two_mode_ffft_qbk,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )
    gold_plaquette_trotter_step = PlaquetteTrotterStep(
        fermionic_swap_qubrick=fswap_block_qbk,
        two_mode_ffft_qubrick=two_mode_ffft_qbk,
        exptXXYY_qubrick=exptXXYY_qbk,
        localize_plaquettes_qubrick=localize_plaquettes_qubrick,
    )

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
    return U_closed_control, U_powered_directional


def baseline_computation(
    *,
    qc: QPU,
    psi_register: Qubits,
    fermi_hubbard_data: FermiHubbardData,
    circuit_data: CircuitData,
) -> None:
    """_summary_

    Args:
        qc (QPU): _description_
        psi_register (Qubits): _description_
        fermi_hubbard_data (FermiHubbardData): _description_
        circuit_data (ErrorBudget): _description_
    """
    time_evolution = build_baseline_qubricks(num_batches=circuit_data.no_batches)

    # Solely for qubit counting
    phase_reg = Qubits(1, "phase_reg", qc)
    phase_reg.identity()

    allocate_catalyst_registers_IPG(
        x_dim=fermi_hubbard_data.x_dim,
        y_dim=fermi_hubbard_data.y_dim,
        potential_coefficient=fermi_hubbard_data.u,
        kinetic_coefficient=fermi_hubbard_data.t,
        total_evolution_time=circuit_data.evolution_time,
        n_trotter_steps=circuit_data.no_trotter_steps,
        n_hwp_batches=circuit_data.no_batches,
        qc=qc,
    )

    # Repeat queries
    if circuit_data.no_queries > 2:
        qc.use_jump_back_iterations = True
        jump_target = qc.jump_back_target(max_num_loops=circuit_data.no_queries - 2)
        time_evolution.compute(psi_register, fermi_hubbard_data, ctrl=0, use_jump_back=False)
        qc.jump_back(jump_target, max_num_loops=circuit_data.no_queries - 2)
        time_evolution.compute(psi_register, fermi_hubbard_data, ctrl=0)
    else:
        for _ in range(circuit_data.no_queries):
            time_evolution.compute(psi_register, fermi_hubbard_data, ctrl=0, use_jump_back=True)


def improved_computation(
    *,
    qc: QPU,
    psi_register: Qubits,
    fermi_hubbard_data: FermiHubbardData,
    circuit_data: CircuitData,
    use_black_box: bool = True,
) -> None:
    """_summary_

    Args:
        qc (QPU): _description_
        psi_register (Qubits): _description_
        phase_register (Qubits): _description_
        fermi_hubbard_data (FermiHubbardData): _description_
        circuit_data (ErrorBudget): _description_
        use_black_box (bool, optional): _description_. Defaults to True.

    """
    n_phase_qubits = int(np.ceil(np.log2(circuit_data.no_queries))) + 1
    phase_register = Qubits(n_phase_qubits, "phase_reg", qc)

    closed_control, powered_directional = build_improved_qubricks(
        num_batches=circuit_data.no_batches, use_black_box=use_black_box
    )

    allocate_catalyst_registers_PIG(
        x_dim=fermi_hubbard_data.x_dim,
        y_dim=fermi_hubbard_data.y_dim,
        potential_coefficient=fermi_hubbard_data.u,
        kinetic_coefficient=fermi_hubbard_data.t,
        total_evolution_time=circuit_data.evolution_time,
        n_trotter_steps=circuit_data.no_trotter_steps,
        n_hwp_batches=circuit_data.no_batches,
        qc=qc,
    )
    cat_qubits = fetch_catalyst_qubits(qc)

    # Call qubricks
    ctrl = ControlQubit(phase_register[0], defer_rotations=True)
    closed_control.compute(psi_register, catalyst_reg=cat_qubits, data=fermi_hubbard_data, ctrl=ctrl)

    if circuit_data.no_queries > 1:
        for j, qbit in enumerate(phase_register[1:]):
            power = 2**j
            qbit = DirectionalControlQubit(qbit, defer_rotations=True)
            powered_directional.compute(
                psi_register,
                fermi_hubbard_data,
                catalyst_reg=cat_qubits,
                power=power,
                ctrl=qbit,
                use_jump_back=use_black_box,
            )
