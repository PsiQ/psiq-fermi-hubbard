"""Qubricks for directional Hamming weight phasing."""

import numpy as np
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, PhasingCircuit, PowerOfTwoBatchedHammingWeightPhasing
from psiqdk.algorithms._subroutines.hamming_weight_phasing import (
    _calculate_adder_segment_resources_compute_and_uncompute,
)
from psiqdk.workbench import Qubits, Qubrick, units
from psiqdk.workbench.utils import is_pow_two

from ..utils.control_qubit import ControlQubit, DirectionalControlQubit


class DirectionalHammingWeightPhasing(Qubrick):
    """Implement a directionally controlled rotation stack using hamming weight phasing.

    This implements the equivalent of a closed control stack of rotations of a specified angle followed
    by an open control stack of rotations by the negative of this angle. Utilised e.g. to implement
    directional phase kickback in QPE on Trotterized time evolution of translationally invariance lattice models, which approximately halves
    the query count.
    """

    def __init__(
        self,
        angle,
        rot_is_rz=False,
        hamming_weight_qubrick=None,
        use_catalyst_state=False,
        catalyst_state_reg=None,
        use_padding=False,
        use_black_box=False,
        preserve_global_phase: bool = True,
        turn_on_cnots: bool = True,
        **kwargs,
    ):
        """Construct the Qubrick.

        Args:
            angle (float): An angle (in degrees) specifying the angle of the rotations
            rot_is_rz (bool): Flag to determine if attempting to perform Rz rotations or phase gates
            hamming_weight_qubrick (Qubrick): A Qubrick that computes the hamming weight of a register
            use_catalyst_state (bool): A flag to determine if the Phasing circuit should be used to synthesize the phase
                gradient.
            catalyst_state_reg (Qubits): The catalyst state for the PhaseC circuit
            use_padding (bool): A flag to determine if the Hamming Weight register should be padded with clean ancilla
                so that the size is consistent with the size of the catalyst state. If it is not padded, then a single
                phase gate needs to be applied during the PhaseC circuit. If it is padded, then the rotations should be
                completely decomposed via Toffolis when adding onto the catalyst state assuming the catalyst state is
                large enough to implement the angle passed in exactly.
            use_black_box (bool): Uses black box AV counts for adders in compute HWP qubrick if set to True. Default is False
            preserve_global_phase (bool): If False, a single Rz fix-up is required. If True we need a phase gate and an Rz.
                                        NOTE: global phase does not need to be preserved for expected functionality
                                        iff no additional control structure
            turn_on_cnots (bool): If True, applies CNOT fanouts on catalyst qubits when directionally controlling
                the Hamming weight phasing. This flag is to remove cancellable CNOTs on catalysts when the directional
                HWP is called multiple times on the same qubits.
            **kwargs (dict[str, Any]): Additonal Qubrick kwargs
        """
        super().__init__(**kwargs)
        self.use_black_box = use_black_box
        if hamming_weight_qubrick is None:
            hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees(use_black_box=self.use_black_box)
        hamming_weight_qubrick.use_black_box = self.use_black_box
        self.angle = angle
        self.hamming_weight_qubrick = hamming_weight_qubrick
        self.use_catalyst_state = use_catalyst_state
        self.preserve_global_phase = preserve_global_phase
        self.rot_is_rz = rot_is_rz
        if not self.rot_is_rz:
            # acquired phase will be local for phase gate
            # as such fix-ups are required and we default to version with minimal cnot fanout
            self.preserve_global_phase = True
        self.turn_on_cnots = turn_on_cnots
        if self.use_catalyst_state:
            self.phasing_circuit = DirectionalPhasingCircuit(
                self.angle, use_black_box=self.use_black_box, turn_on_cnots=turn_on_cnots
            )
        self.catalyst_state_reg = catalyst_state_reg
        self.use_padding = use_padding

    def _compute(
        self,
        target_register: Qubits,
        ctrl: Qubits,
    ):
        """Implement a directionally controlled stack of rotations using hamming weight phasing.

        Args:
            target_register (Qubits): The register on which to apply the rotations
            ctrl (Qubits): Directional control.

        Raises:
            ValueError: If no control qubit provided or the control has more than 1 qubit.
        """
        if not (isinstance(ctrl, Qubits) and ctrl.num_qubits == 1):
            raise ValueError(
                "DirectionalHammingWeightPhasing requires a single control qubit. To use without a control call HammingWeightPhasing."
            )

        self.hamming_weight_qubrick.compute(target_register)
        hamming_weight_register = self.hamming_weight_qubrick.get_result_qreg()

        if self.use_catalyst_state:
            # Accumulated fixup angle from decomposing two directionally
            # controlled phase gates into one uncontrolled phase gate conjugated
            # by open-controlled CNOTs
            payload_rot_fixup = (1 << hamming_weight_register.num_qubits) * self.angle

            size_of_padding = len(self.catalyst_state_reg) - len(hamming_weight_register)
            if self.use_padding and (size_of_padding > 0):
                padding = self.alloc_temp_qreg(size_of_padding, "hw_padding")
                combined_reg = hamming_weight_register | padding
                self.phasing_circuit.compute(
                    combined_reg[:-3],
                    self.catalyst_state_reg[:-3],
                    ctrl=ctrl,
                    final_qubits=combined_reg[-3:],
                )
                padding.write(0)
                padding.release()
            else:
                self.phasing_circuit.compute(hamming_weight_register, self.catalyst_state_reg, ctrl=ctrl)
        else:
            payload_rot_fixup = sum((1 << i) * self.angle for i in range(hamming_weight_register.num_qubits))

            padding = 0
            if self.use_padding:
                size_of_padding = 0
                current_angle = self.angle % 360
                while current_angle:
                    current_angle = self.angle * (1 << size_of_padding)
                    current_angle = current_angle % 360
                    if current_angle != 0:
                        size_of_padding += 1

                padding = self.alloc_temp_qreg(size_of_padding, "hw_padding")
            padded_hw_reg = hamming_weight_register | padding

            # log-sized CNOT fanout
            padded_hw_reg.x(~ctrl)

            for ancilla_index in range(len(padded_hw_reg)):
                current_angle = self.angle * (1 << ancilla_index)
                current_angle = current_angle % 360
                if current_angle == 315:
                    padded_hw_reg[ancilla_index].t_inv()
                elif current_angle == 270:
                    padded_hw_reg[ancilla_index].s_inv()
                elif current_angle != 0:
                    padded_hw_reg[ancilla_index].phase(current_angle)

            # log CNOT fanout
            padded_hw_reg.x(~ctrl)

            if not isinstance(padding, int):
                padding.write(0)
                padding.release()
        self.hamming_weight_qubrick.uncompute()

        # whether phase or Rz, we need to correct for bi-controlled phase gate -> (cnot * uncontrolled phase gate * cnot)
        # but, if we're not preserving global phase we can combine this fixup with that required when rot_is_rz is True (continues below)
        if self.preserve_global_phase:
            (~ctrl).reflect(theta=-1 * payload_rot_fixup)
        # if rot_is_rz is True, we also need to correct for implementing via phase gate(s)
        if self.rot_is_rz:
            rot_tower_fixup = -(self.angle / 2) * len(target_register)
            if self.preserve_global_phase:
                ctrl.rz(theta=2 * rot_tower_fixup)  # Rz -> phase gate(s)
            else:
                ctrl.rz(
                    theta=2 * rot_tower_fixup + payload_rot_fixup
                )  # combined fix-ups: Rz -> phase and bi-ctrl phase -> uncontrolled phase + cnot's


class DirectionalPhasingCircuit(PhasingCircuit):
    """Implement the phasing circuit with directional control."""

    def __init__(self, base_angle, use_black_box, rot_is_rz=False, turn_on_cnots=True, **kwargs):
        """Construct the directional phasing circuit qubrick for a given angle and register size.

        Args:
            base_angle (float): An angle (in degrees) specifying the base angle of the growing tower of rotations
            use_black_box (bool): Uses black box AV counts if set to True. Default is False
            rot_is_rz (bool): Flag to determine if attempting to perform Rz rotations or phase gates
            turn_on_cnots (bool): If True, applies CNOT fanouts on catalyst qubits when directionally controlling
                the Hamming weight phasing. This flag is to remove cancellable CNOTs on catalysts when the directional
                HWP is called multiple times on the same qubits.
            **kwargs (dict[str, Any]): Additonal Qubrick kwargs

        Notes:
            - Qubrick brings a global phase difference from directionally controlling the payload phase rotation,
              i.e. CNOT * payload_phase_rot * CNOT. This fix-up in this case is (~ctrl).reflect(-final_angle) where
              final_angle is defined in the code.
            - Additionally, if we want to synthesize a directionally controlled phase gradients that are of the RZ type,
              we need an additional phase fix-up of ctrl.rz(-n_targ*angle) as the phase catalysis circuit (this qubrick)
              is defined in terms of phase rotations.
            - Both of these fix-ups are handled in directionalHammingWeightPhasing qubrick. As such, we do not need
              to check that the rotation type of the phase gradient is RZ in this qubrick.
        """
        super().__init__(base_angle=base_angle, rot_is_rz=rot_is_rz, use_black_box=use_black_box, **kwargs)
        self.counter = 0
        self.turn_on_cnots = turn_on_cnots

    def _compute(self, target_reg: Qubits, catalyst_reg: Qubits, ctrl: Qubits, final_qubits: Qubits | None = None):
        """Use the phasing circuit to implement a tower of rotations with a directional control.

        Args:
            target_reg (Qubits): The state to implement the rotations upon
            catalyst_reg (Qubits): The catalyst state to use which is specific to the base angle being implemented
            ctrl (Qubits): directional control
            final_qubits (Qubits): If passed in, these are the three qubits on which we just perform Z, S, and T
                directly instead of adding on to the catalyst state

        Raises:
            ValueError: If target register is too large for the cataylst register.
            ValueError: If no control qubit provided or the control has more than 1 qubit.

        Notes:
            - The implemented circuit applies rotations of base_angle*(2**i) for the i'th qubit of
                the target_reg where i runs from 0 to len(target_reg) - 1.
        """
        if not (isinstance(ctrl, Qubits) and ctrl.num_qubits == 1):
            raise ValueError(
                "DirectionalPhasingCircuit requires a single control qubit. To use without a control call PhasingCircuit."
            )

        if not (len(target_reg) <= len(catalyst_reg)):
            raise ValueError("The target reg is too large for the supplied catalyst reg!")
        num_padding_qubits_needed = len(catalyst_reg) - len(target_reg) - 1
        if num_padding_qubits_needed > 0:
            target_reg_padding = self.alloc_temp_qreg(num_padding_qubits_needed, "phasec_padding")
            target_reg = target_reg | target_reg_padding
        anc = self.alloc_temp_qreg(1, f"phasec_{self.counter}")

        if self.turn_on_cnots:
            catalyst_reg.x(~ctrl)

        anc.lelbow(target_reg[0] | (catalyst_reg[0] == 0))
        num_remaining_levels = len(target_reg) - 1
        ancillae = [anc]

        while num_remaining_levels > 0:
            top = ancillae[-1]
            mid = target_reg[len(target_reg) - num_remaining_levels]
            bottom = catalyst_reg[len(target_reg) - num_remaining_levels]

            anc = self._compute_subblock(top, mid, bottom)
            ancillae.append(anc)
            num_remaining_levels -= 1

        if final_qubits is not None:
            final_qubits[0].t_inv()
            final_qubits[1].s_inv()
            final_qubits[2].z()

        final_angle = (1 << len(target_reg)) * self.base_angle
        final_angle = final_angle % 360
        (ancillae[-1]).x(~ctrl)
        if final_angle == 315 * units.deg:
            ancillae[-1].t_inv(cond=ctrl)
        elif final_angle == 270 * units.deg:
            ancillae[-1].s_inv(cond=ctrl)
        elif final_angle != 0 * units.deg:
            (ancillae[-1]).phase(final_angle)
        (ancillae[-1]).x(~ctrl)

        while len(ancillae) > 1:
            top = ancillae[-2]
            mid = target_reg[len(ancillae) - 1]
            bottom = catalyst_reg[len(ancillae) - 1]
            self._uncompute_subblock(top, mid, bottom, ancillae[-1], ctrl)
            ancillae.pop(-1)

        ancillae[-1].relbow(target_reg[0] | (catalyst_reg[0] == 0))
        ancillae[-1].release()

        catalyst_reg[0].x(cond=(target_reg[0]))

        if num_padding_qubits_needed > 0:
            target_reg_padding.write(0)
            target_reg_padding.release()

        if self.turn_on_cnots:
            catalyst_reg.x(~ctrl)

    def _uncompute_subblock(self, top, mid, bottom, anc, ctrl):
        """Uncompute half-adder-like block."""
        if self.use_black_box:
            cost = _calculate_adder_segment_resources_compute_and_uncompute(top, mid, bottom, anc, ctrl)
            self.get_qc().add_cost_event(cost)
            (top | mid | bottom | anc).box()
            anc.release()  # or else qubit count will artificially go up
        else:
            anc.x(top)
            anc.relbow(mid | (bottom == 0))
            anc.release()
            bottom.x(mid)  # NOTE: Not controlled compared to regular PhasingCircuit
            (mid | bottom).x(top)


class PowerOfTwoBatchedDirectionalHammingWeightPhasing(PowerOfTwoBatchedHammingWeightPhasing):
    """Implements directional Hamming weight phasing by splitting the target register into batches.

    This Qubrick divides a large directional rotation tower into smaller batches, where the number of batches
    is a power of two. Each batch is processed with a smaller directional Hamming weight phasing circuit.
    This approach can significantly reduce the number of qubits and catalyst rotation
    requirements for large registers.

    This is particularly useful for applications like Fermi-Hubbard simulation where the number of
    parallel Rz rotations is either L^2/2 or L^2, where L is the lattice size.

    Note that depending on the type of control, this qubrick implements standard closed control
    Hamming weight phasing or directionally controlled Hamming weight phasing.
    """

    def __init__(
        self,
        hamming_weight_qubrick: Qubrick | None = None,
        n_hwp_batches: int = 1,
        rot_is_rz: bool = True,
        use_black_box: bool = False,
        preserve_global_phase: bool = False,
        turn_on_cnots: bool = True,
        **kwargs,
    ):
        """Constructs a batched directional Hamming weight phasing Qubrick.

        Args:
            hamming_weight_qubrick: Qubrick for computing the Hamming weight for
                the Hamming weight phasing operation. If None, a default method using
                ComputeHammingWeightGroupOfThrees with GidneyAdd will be used.
            n_hwp_batches: Number of batches to split the rotations into. Must be a power of 2.
                Larger values reduce resource costs but may increase circuit depth.
            rot_is_rz: Whether to use RZ rotations (True) or phase rotations (False).
            use_black_box (bool): Uses black box AV counts if set to True. Default is False.
            preserve_global_phase (bool): Include phase fix-ups when using Rz rotations such that global phase is preserved.
                                        NOTE: global phase does not need to be preserved for expected functionality
            turn_on_cnots (bool): If True, applies CNOT fanouts on catalyst qubits when directionally controlling
                the Hamming weight phasing. This flag is to remove cancellable CNOTs on catalysts when the directional
                HWP is called multiple times on the same qubits.
            **kwargs (dict[str, Any]): Additional arguments to pass to the Qubrick constructor.

        Raises:
            ValueError: If n_hwp_batches is not a power of 2.
        """
        if hamming_weight_qubrick is None:
            self._hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
        else:
            self._hamming_weight_qubrick = hamming_weight_qubrick

        if is_pow_two(n_hwp_batches):
            self.n_hwp_batches = n_hwp_batches
        else:
            raise ValueError("Number of HWP batches must be a power of 2.")
        self.preserve_global_phase = preserve_global_phase
        self.rot_is_rz = rot_is_rz
        self.use_black_box = use_black_box
        self.turn_on_cnots = turn_on_cnots

        super().__init__(
            hamming_weight_qubrick=self._hamming_weight_qubrick,
            n_hwp_batches=n_hwp_batches,
            rot_is_rz=self.rot_is_rz,
            use_black_box=self.use_black_box,
            **kwargs,
        )

    def _compute_standard(self, rz_rotation_angle: float, target_register: Qubits, ctrl: int | ControlQubit):
        """Compute batched Hamming weight phasing operation assuming no or standard closed control.

        Args:
            rz_rotation_angle: Rotation angle (in degrees) for the phase operations.
            target_register: Register on which to apply the rotations.
            ctrl (int or ControlQubit): Quantum control (standard)
        """
        if not (ctrl == 0 or isinstance(ctrl, ControlQubit)):
            raise ValueError("ctrl for standard compute must be 0 or ControlQubit.")
        super()._compute(rz_rotation_angle, target_register, ctrl)

    def _compute_directional(self, rz_rotation_angle: float, target_register: Qubits, ctrl: DirectionalControlQubit):
        """Compute batched directional Hamming weight phasing operation.

        This method splits the target register into batches, then applies a directional Hamming weight
        phasing operation to each batch separately. This approach can significantly reduce
        Toffoli gate counts and catalyst rotation requirements for large registers.

        Args:
            rz_rotation_angle: Rotation angle (in degrees) for the phase operations.
            target_register: Register on which to apply the rotations.
            ctrl (DirectionalControlQubit): directional control qubit

        Raises:
            ValueError: If no control qubit provided or the control has more than 1 qubit.
            ValueError: If the number of batches is greater than the number of qubits
                in the target register.
        """
        if not (isinstance(ctrl, DirectionalControlQubit) and ctrl.num_qubits == 1):
            raise ValueError("Directional Phasing requires a single control qubit.")

        if self.n_hwp_batches > target_register.num_qubits:
            raise ValueError("Number of HWP batches must be less than the size of the target register.")

        qc = self.get_qc()
        if self.n_hwp_batches > 1:
            list_indices = np.array_split(np.arange(target_register.num_qubits), self.n_hwp_batches)
            rotation_towers = [target_register[ind] for ind in list_indices]
        else:
            rotation_towers = [target_register]

        size_of_catalyst_state = max(1, int(np.floor(np.log2(rotation_towers[0].num_qubits))) + 1)
        cat_angle = (1 << (size_of_catalyst_state - 1)) * rz_rotation_angle
        with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg:
            hwp = DirectionalHammingWeightPhasing(
                angle=rz_rotation_angle,
                hamming_weight_qubrick=self.hamming_weight_qubrick,
                use_catalyst_state=True,
                catalyst_state_reg=catalyst_state_reg,
                rot_is_rz=self.rot_is_rz,
                use_black_box=self.use_black_box,
                preserve_global_phase=self.preserve_global_phase,
                turn_on_cnots=self.turn_on_cnots,
            )
            for i in range(self.n_hwp_batches):
                hwp.compute(rotation_towers[i], ctrl=ctrl)

        self.set_result_qreg(catalyst_state_reg, "catalyst_reg")

    def _compute(self, rz_rotation_angle: float, target_register: Qubits, ctrl: Qubits | int):
        if isinstance(ctrl, DirectionalControlQubit):
            self._compute_directional(rz_rotation_angle, target_register, ctrl)
        else:
            self._compute_standard(rz_rotation_angle, target_register, ctrl)
