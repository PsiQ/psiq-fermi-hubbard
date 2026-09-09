"""Qubricks for implementing the optimized Fermi-Hubbard Trotterizaton circuit in arxiv:2012.09238."""

from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees
from psiqdk.workbench import Qubits, Qubrick

from ..utils.control_qubit import ControlQubit, DirectionalControlQubit
from .directional_hamming_weight_phasing import PowerOfTwoBatchedDirectionalHammingWeightPhasing
from .fermi_hubbard_data import InteractionTermData, PlaquetteTermData
from .hopping import PlaquetteTrotterStep, exptXXYYViaPPR
from .interaction import InteractionTrotterStep


class HubbardPlaquetteTrotterizationIPG(Qubrick):
    """Implements the time-evolution :math:`exp(itH))` of the Fermi-Hubbard Hamiltonian based on arxiv:2012.09238, with Trotter ordering [Interaction, Pink, Gold] so that the interaction term is merged between steps."""

    def __init__(
        self,
        interaction_trotter_step: Qubrick | None = None,
        pink_plaquette_trotter_step: Qubrick | None = None,
        gold_plaquette_trotter_step: Qubrick | None = None,
        **kwargs,
    ):
        """Construct the HubbardPlaquetteTrotterization Qubrick.

        Args:
            interaction_trotter_step: qubrick that implements time evolution of the Interaction terms in the Hubbard model - exp(itH_I) - based on 2012.09238.
            pink_plaquette_trotter_step: qubrick that implements time evolution for a pink plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            gold_plaquette_trotter_step: qubrick that implements time evolution for a gold plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            **kwargs: Other arguments to pass to the init.
        """
        super().__init__(**kwargs)

        if interaction_trotter_step is None:
            self.interaction_trotter_step = InteractionTrotterStep()
        else:
            self.interaction_trotter_step = interaction_trotter_step

        if pink_plaquette_trotter_step is None:
            self.pink_plaquette_trotter_step = PlaquetteTrotterStep()
        else:
            self.pink_plaquette_trotter_step = pink_plaquette_trotter_step

        if gold_plaquette_trotter_step is None:
            self.gold_plaquette_trotter_step = PlaquetteTrotterStep()
        else:
            self.gold_plaquette_trotter_step = gold_plaquette_trotter_step

    def _compute(self, target_reg, data, catalyst_reg=None, power=1, ctrl=0, use_jump_back=False):
        """Compute the Plaquette Trotter step.

        Args:
            target_reg (Qubits): Qubit register storing the system. Register is of size ``number_of_spin_orbitals`` or
                ``2*x_dimension*y_dimension``
            data (Vanilla2DFermiHubbardData): Fermi-Hubbard system dataclass.
            catalyst_reg (Qubits): Qubit register storing all the catalyst qubits for I, P, G terms.
            power (int): j in U^{j} for U = Trotterized time evolution operator.
            ctrl (int or Qubits, optional): The quantum controls that control the action of the Qubrick.
            use_jump_back (bool): If True, uses the jump functionality to speed up resource counting. Default is False.

        Note:
            - See Eq. (E2) in arxiv:2012.09238
        """
        if catalyst_reg is not None:
            raise ValueError("This input arg does not do anything. We added to have consistent compute args.")

        single_trotter_step_evolution_time = data.total_evolution_time / (data.n_trotter_steps)
        # First extract data to create data classes for interaction term data
        interaction_coefficient = data.u / 4
        # Implicitly assumes IPGPI second order trotter so the first and last interaction evolutions are by time t/2 whereas when we have sequential trotter steps two interaction steps can be merged together to make a single evolution by time t.
        interaction_data = InteractionTermData(
            data.enumeration,
            single_trotter_step_evolution_time / 2,
            interaction_coefficient,
            particle_hole_symmetry=data.particle_hole_symmetry,
        )

        # Then extract data to create data classes for plaquette term data
        # Implicitly assumes IPGPI second order trotter so that the pink evolution is t/2 and the gold is two merged evolutions to make t.
        pink_plaquette_data = PlaquetteTermData(
            "pink", data.enumeration, single_trotter_step_evolution_time / 2, data.t
        )
        gold_plaquette_data = PlaquetteTermData("gold", data.enumeration, single_trotter_step_evolution_time, data.t)

        for i in range(power):
            # Initial I half-step
            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)

            interaction_data.evolution_time *= 2  # squish neighboring interaction steps together

            total_trotter_steps = data.n_trotter_steps
            if use_jump_back and total_trotter_steps > 2:
                self.get_qc().use_jump_back_iterations = True
                jump_target = self.get_qc().jump_back_target(max_num_loops=total_trotter_steps - 2)
                self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)
                self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)
                self.get_qc().jump_back(jump_target, max_num_loops=total_trotter_steps - 2)

                self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)
                self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                # Final I should be a half-step, not a merged full-step
                interaction_data.evolution_time /= 2
                self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)

            else:
                # Perform
                for trotter_step in range(data.n_trotter_steps):
                    if trotter_step == data.n_trotter_steps - 1:
                        # For the last trotter step, interaction term is back to normal amount of time (t/2)
                        interaction_data.evolution_time /= 2

                    # Pink
                    self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                    # Gold
                    self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)
                    # Pink
                    self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
                    # Interaction
                    self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)


class HubbardPlaquetteTrotterizationPIG(Qubrick):
    """Implements the time-evolution :math:`exp(itH))` of the Fermi-Hubbard Hamiltonian based on arxiv:2012.09238, with Trotter ordering [Pink, Interaction, Gold] so that pink plaquettes are merged between steps."""

    def __init__(
        self,
        interaction_trotter_step: Qubrick | None = None,
        pink_plaquette_trotter_step: Qubrick | None = None,
        gold_plaquette_trotter_step: Qubrick | None = None,
        **kwargs,
    ):
        """Construct the HubbardPlaquetteTrotterization Qubrick.

        Args:
            interaction_trotter_step: qubrick that implements time evolution of the Interaction terms in the Hubbard model - exp(itH_I) - based on 2012.09238.
            pink_plaquette_trotter_step: qubrick that implements time evolution for a pink plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            gold_plaquette_trotter_step: qubrick that implements time evolution for a gold plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            **kwargs: Other arguments to pass to the init.

        Note:
            - This qubrick can be used for closed controlled U, but this would be less efficient.
        """
        super().__init__(**kwargs)

        if interaction_trotter_step is None:
            self.interaction_trotter_step = InteractionTrotterStep()
        else:
            self.interaction_trotter_step = interaction_trotter_step

        if pink_plaquette_trotter_step is None:
            self.pink_plaquette_trotter_step = PlaquetteTrotterStep()
        else:
            self.pink_plaquette_trotter_step = pink_plaquette_trotter_step

        if gold_plaquette_trotter_step is None:
            self.gold_plaquette_trotter_step = PlaquetteTrotterStep()
        else:
            self.gold_plaquette_trotter_step = gold_plaquette_trotter_step

    def _compute(
        self,
        target_reg,
        data,
        catalyst_reg=None,
        power: int = 1,
        ctrl: int | DirectionalControlQubit = 0,
        use_jump_back=False,
    ):
        """Compute the Plaquette Trotter step raised to a power.

        Args:
            target_reg (Qubits): Qubit register storing the system. Register is of size ``number_of_spin_orbitals`` or
                ``2*x_dimension*y_dimension``
            data (Vanilla2DFermiHubbardData): Fermi-Hubbard system dataclass.
            catalyst_reg (Qubits): Qubit register storing all the catalyst qubits for P, I, G terms.
            power (int): j in U^{j} for U = Trotterized time evolution operator.
            ctrl (int or ControlQubit or DirectionalControlQubit, optional): The quantum controls that control
                the action of the Qubrick.
                If ctrl is 0, no control qubit.
                If ctrl is of DirectionalControlQubit type, then we implement the directionally controlled
                    evolution.
            use_jump_back (bool): If True, uses the jump functionality to speed up resource counting. Default is False.

        Note:
            - See Eq. (E2) in arxiv:2012.09238
        """
        if power < 1:
            raise ValueError("Input variable 'power' must be a positive integer.")

        if not (ctrl == 0 or isinstance(ctrl, DirectionalControlQubit)):
            raise ValueError(
                "For closed controlled evolution, a more efficient implementation would be HubbardPlaquetteTrotterizationPIGClosedControl."
            )

        single_trotter_step_evolution_time = data.total_evolution_time / data.n_trotter_steps

        interaction_coeff_term = data.u / 4  # divide by four from Jordan Wigner

        interaction_data = InteractionTermData(
            data.enumeration,
            single_trotter_step_evolution_time / 2,
            interaction_coeff_term,
            particle_hole_symmetry=data.particle_hole_symmetry,
        )

        pink_plaquette_data = PlaquetteTermData(
            "pink",
            data.enumeration,
            single_trotter_step_evolution_time / 2,
            data.t,
        )

        gold_plaquette_data = PlaquetteTermData(
            "gold",
            data.enumeration,
            single_trotter_step_evolution_time,
            data.t,
        )

        total_trotter_steps = power * data.n_trotter_steps

        if isinstance(ctrl, DirectionalControlQubit):
            # Assuming pink and gold terms also use the same batched HWP qubrick instantiation
            intermediate_cnots_on = self.interaction_trotter_step.batched_hamming_weight_phasing_qubrick.turn_on_cnots

            if not intermediate_cnots_on:
                catalyst_reg.x(~ctrl)

        # Initial P half-step
        self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)

        # Interior pink steps are doubled because adjacent P half-steps merge
        pink_plaquette_data.evolution_time *= 2

        # use_jump_back is an optimization where we make a loop in the op stream.
        if use_jump_back and total_trotter_steps > 2:
            self.get_qc().use_jump_back_iterations = True
            jump_target = self.get_qc().jump_back_target(max_num_loops=total_trotter_steps - 2)
            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)
            self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)
            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)
            self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)
            self.get_qc().jump_back(jump_target, max_num_loops=total_trotter_steps - 2)

            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)
            self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)
            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)
            # Final P should be a half-step, not a merged full-step
            pink_plaquette_data.evolution_time /= 2
            self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)

        else:
            for trotter_step in range(total_trotter_steps):
                is_last_step = trotter_step == total_trotter_steps - 1

                if is_last_step:
                    # Final P should be a half-step, not a merged full-step
                    pink_plaquette_data.evolution_time /= 2

                self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)

                self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)

                self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)

                self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)

        if isinstance(ctrl, DirectionalControlQubit):
            if not intermediate_cnots_on:
                catalyst_reg.x(~ctrl)
            ctrl.resolve_rotations()


class HubbardPlaquetteTrotterizationPIGClosedControl(Qubrick):
    """Implements a closed-controlled time-evolution :math:`exp(itH))` of the Fermi-Hubbard Hamiltonian based on arxiv:2012.09238,
    with Trotter ordering [Pink, Interaction, Gold] so that pink plaquettes are merged between steps. This subroutine can be
    used on the first phase qubit of a QPE employing directional phase kickback.
    """

    def __init__(
        self,
        interaction_trotter_step: Qubrick | None = None,
        pink_plaquette_trotter_step: Qubrick | None = None,
        gold_plaquette_trotter_step: Qubrick | None = None,
        **kwargs,
    ):
        """Construct the HubbardPlaquetteTrotterization Qubrick.

        Args:
            interaction_trotter_step: qubrick that implements time evolution of the Interaction terms in the Hubbard model - exp(itH_I) - based on 2012.09238.
            pink_plaquette_trotter_step: qubrick that implements time evolution for a pink plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            gold_plaquette_trotter_step: qubrick that implements time evolution for a gold plaquette exp(itK). See Equation E10 of https://arxiv.org/abs/2012.09238.
            **kwargs: Other arguments to pass to the init.
        """
        super().__init__(**kwargs)

        if interaction_trotter_step is None:
            hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
            hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
                hamming_weight_qubrick,
                use_black_box=False,
            )
            self.interaction_trotter_step = InteractionTrotterStep(hwp_qbk)
        else:
            self.interaction_trotter_step = interaction_trotter_step

        if pink_plaquette_trotter_step is None:
            hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
            hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
                hamming_weight_qubrick,
                use_black_box=False,
            )
            exptXXYY_qbk = exptXXYYViaPPR(hwp_qbk, use_black_box=False)
            self.pink_plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qbk)
        else:
            self.pink_plaquette_trotter_step = pink_plaquette_trotter_step

        if gold_plaquette_trotter_step is None:
            hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
            hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
                hamming_weight_qubrick,
                use_black_box=False,
            )
            exptXXYY_qbk = exptXXYYViaPPR(hwp_qbk, use_black_box=False)
            self.gold_plaquette_trotter_step = PlaquetteTrotterStep(exptXXYY_qubrick=exptXXYY_qbk)
        else:
            self.gold_plaquette_trotter_step = gold_plaquette_trotter_step

    def _compute(self, target_reg, catalyst_reg, data, ctrl: ControlQubit):
        """Compute the Plaquette Trotter step.

        Args:
            target_reg (Qubits): Qubit register storing the system. Register is of size ``number_of_spin_orbitals`` or
                ``2*x_dimension*y_dimension``
            catalyst_reg (Qubits): Qubit register storing all the catalyst qubits for P, I, G terms.
            data (Vanilla2DFermiHubbardData): Fermi-Hubbard system dataclass.
            ctrl (ControlQubit): The quantum controls that control the action of the Qubrick.

        Note:
            - See Eq. (E2) in arxiv:2012.09238
        """
        if not (isinstance(ctrl, ControlQubit) or isinstance(ctrl, Qubits)):
            raise ValueError("This qubrick is optimized for closed controlled evolution operator.")

        single_trotter_step_evolution_time = data.total_evolution_time / (data.n_trotter_steps)

        # First extract data to create data classes for interaction term data
        interaction_coeff_term = data.u / 4  # divide by four from Jordan Wigner
        interaction_data = InteractionTermData(
            data.enumeration,
            single_trotter_step_evolution_time / 2,
            interaction_coeff_term,
            particle_hole_symmetry=data.particle_hole_symmetry,
        )

        # Then extract data to create data classes for plaquette term data
        pink_plaquette_data = PlaquetteTermData(
            "pink", data.enumeration, single_trotter_step_evolution_time / 2, data.t
        )
        gold_plaquette_data = PlaquetteTermData("gold", data.enumeration, single_trotter_step_evolution_time, data.t)

        self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=0)

        pink_plaquette_data.evolution_time *= 2  # merging neighbouring plaquette evolution steps together

        half_steps = data.n_trotter_steps // 2
        is_even = data.n_trotter_steps % 2 == 0

        # First half of the evolutions. This is uncontrolled.
        for trotter_step in range(half_steps):
            is_middle_pink = is_even and trotter_step == (half_steps - 1)

            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=0)

            self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=0)

            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=0)

            pink_ctrl = ControlQubit(ctrl, defer_rotations=True) if is_middle_pink else 0

            self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=pink_ctrl)

        # Assuming pink and gold terms also use the same batched HWP qubrick instantiation
        intermediate_cnots_on = self.interaction_trotter_step.batched_hamming_weight_phasing_qubrick.turn_on_cnots

        # Prepare control for second half
        if is_even:
            # Case: pink was the center/closed controlled term
            ctrl = DirectionalControlQubit.from_control(pink_ctrl)
            int_ctrl = ctrl  # including interaction term
            if not intermediate_cnots_on:
                catalyst_reg.x(~ctrl)
        else:
            # Case: the next gold will be the center
            # Set control value for next interaction term (just before the center gold term)
            int_ctrl = 0

        # If number of Trotter steps is even, we now move onto directionally controlled terms
        # If number of Trotter steps is odd, we still have to apply one more interaction term
        #    that is uncontrolled before the center gold term that is closed controlled followed
        #    by directionally controlled terms
        for trotter_step in range(half_steps, data.n_trotter_steps):
            is_first_second_half_step = trotter_step == half_steps
            is_last_step = trotter_step == data.n_trotter_steps - 1

            if is_last_step:
                pink_plaquette_data.evolution_time /= 2

            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=int_ctrl)

            if not is_even and is_first_second_half_step:
                # Apply center/closed controlled gold term
                ctrl = ControlQubit(ctrl, defer_rotations=True)
                self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)

                # Now following terms are directionally controlled
                ctrl = DirectionalControlQubit.from_control(ctrl)
                int_ctrl = ctrl  # now, all following interaction term controls are directional
                if not intermediate_cnots_on:
                    catalyst_reg.x(~ctrl)
            else:
                self.gold_plaquette_trotter_step.compute(target_reg, gold_plaquette_data, ctrl=ctrl)

            self.interaction_trotter_step.compute(target_reg, interaction_data, ctrl=ctrl)

            self.pink_plaquette_trotter_step.compute(target_reg, pink_plaquette_data, ctrl=ctrl)

        if not intermediate_cnots_on:
            catalyst_reg.x(~ctrl)
        ctrl.resolve_rotations(merge_phase_and_rz=True)  # Merge Rz and phase
