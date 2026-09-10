"""Qubricks for implementing the hopping term in the Fermi-Hubbard Trotterizaton circuit in arxiv:2012.09238."""

import numpy as np
import psiqdk.workbench.opcodes as opc
from psiqdk.workbench import Qubits, Qubrick
from psiqdk.workbench import units
from psiqdk.workbench.experimental.active_volume_estimation import op_av_lookup_table
from psiqdk.workbench.experimental.symbolics import QubrickCosts
from psiqdk.workbench.ops import QPU_op

from .fermi_hubbard_data import PlaquetteTermData
from .fswapping.fermionic_swap import FermionicSwapWithReplace
from .fswapping.fswap_network import NaivefSWAPNetworkWithReplace


class PlaquetteTrotterStep(Qubrick):
    """Time evolution for a plaquette exp(itK) for plaquette operator K.

    Note:
        - See Equation E10 of https://arxiv.org/abs/2012.09238
    """

    def __init__(
        self,
        fermionic_swap_qubrick: Qubrick | None = None,
        two_mode_ffft_qubrick: Qubrick | None = None,
        exptXXYY_qubrick: Qubrick | None = None,
        localize_plaquettes_qubrick: Qubrick | None = None,
        **kwargs,
    ):
        """Initialize Trotter step.

        Args:
            fermionic_swap_qubrick: qubrick that implements a single fermionic swap (swapping two fermions).
            two_mode_ffft_qubrick: qubrick that implements the two-mode FFFT, based on 2012.09238 (eqn E11).
            exptXXYY_qubrick: qubrick that implements exptXXYY, based on 2012.09238 (eqn E13).
            localize_plaquettes_qubrick: qubrick that implements a fermionic swap network to localize plaquettes.
            **kwargs: Other arguments to pass to the init.

        Note:
            - Angle will be converted to degrees.
            - If exptXXYY_qubrick is None, then the associated rotations are implemented directly without hamming weight phasing.
        """
        super().__init__(**kwargs)

        if fermionic_swap_qubrick is None:
            self.fermionic_swap_qubrick = FermionicSwapWithReplace()
        else:
            self.fermionic_swap_qubrick = fermionic_swap_qubrick

        if two_mode_ffft_qubrick is None:
            self.two_mode_ffft_qubrick = TwoModeFFFTViaControlledHad()
        else:
            self.two_mode_ffft_qubrick = two_mode_ffft_qubrick

        if exptXXYY_qubrick is None:
            self.exptXXYY_qubrick = exptXXYY()
        else:
            self.exptXXYY_qubrick = exptXXYY_qubrick

        if localize_plaquettes_qubrick is None:
            self.localize_plaquettes_qubrick = NaivefSWAPNetworkWithReplace(self.fermionic_swap_qubrick)
        else:
            self.localize_plaquettes_qubrick = localize_plaquettes_qubrick

    def _compute(self, target_reg: Qubits, data: PlaquetteTermData, ctrl: int | Qubits = 0):
        """Compute the time-evolution of all plaquettes grouped by the plaquette indices provided in the dataclass.

        Args:
            target_reg: Qubit register storing the system. The size of the register is the number of spin orbitals or
                2 * x_dimension * y_dimension.
            data: Plaquette term dataclass.
            ctrl: The quantum controls that control the action of the Qubrick.
        """
        plaquette_indices = data.plaquette_indices
        evolution_time = data.evolution_time
        coefficient = data.coefficient

        if plaquette_indices == []:
            return

        # Sign: angle is positive because the sign of the kinetic term is assumed to be negative
        # and the sign convention of Rz is opposite to that of PPRs.
        # Factor of 2: from Rz(theta) having diagonal elements
        # of exp(-i (theta/2)) and exp(i (theta/2)) where as PPR (including
        # for XX+YY) is defined without this factor of 2, as exp(i theta (XX+YY))

        rz_rotation_angle = 1 * 2 * evolution_time * coefficient * 180 / np.pi

        # Relabel fermionic sites so that all plaquettes are localized
        self.localize_plaquettes_qubrick.compute(target_reg, data.enumeration, data.color)
        localized_plaquette_list = self.localize_plaquettes_qubrick.get_classical_result("localized_plaquettes")

        # Collate the site indices to pass into Fermionic swap (separated out for testing - more circuit efficient to localize to this arrangement).
        sites = [1, 2, 3, 4]
        site_indices: dict[int, Qubits] = dict(
            zip(
                sites,
                [list(col) for col in list(zip(*localized_plaquette_list))],
            )
        )

        # Gather qubits into sites for executing plaquette evolution.
        # This is to reduce the depth of the circuit when we're performing operations in parallel across different
        # plaquettes - both for the purpose of the circuit drawings being shorter and easier to read, but also for
        # the use of qubricks like hamming weight phasing.
        # As an example, if we had the following passed in: [[0, 1, 2, 3], [4, 5, 6, 7]], this loop would create
        #   the following:
        #       site_1_qubits = 0 | 4
        #       site_2_qubits = 1 | 5
        #       site_3_qubits = 2 | 6
        #       site_4_qubits = 3 | 7
        site_qubits = {site: target_reg[site_indices[site][0]] for site in sites}
        for q in range(1, len(site_indices[1])):
            for site in sites:
                site_qubits[site] = site_qubits[site] | target_reg[site_indices[site][q]]

        self.fermionic_swap_qubrick.compute(target_reg, site_indices[2], site_indices[3], dagger=True)
        self.two_mode_ffft_qubrick.compute(site_qubits[3], site_qubits[4], dagger=True)
        self.two_mode_ffft_qubrick.compute(site_qubits[2], site_qubits[1], dagger=True)
        self.exptXXYY_qubrick.compute(site_qubits[2], site_qubits[3], rz_rotation_angle, ctrl)
        self.two_mode_ffft_qubrick.uncompute()
        self.two_mode_ffft_qubrick.uncompute()
        self.fermionic_swap_qubrick.uncompute()

        # Uncompute localising fSWAP network
        self.localize_plaquettes_qubrick.uncompute()


class exptXXYY(Qubrick):
    """Implement the operator for exp(i*τ*X_A X_B)*exp(i*τ*Y_A Y_B) evolution
    on qubits A and B.
    """

    def __init__(
        self,
        batched_hamming_weight_phasing_qubrick: Qubrick | None = None,
        **kwargs,
    ):
        """Initialize qubrick implementing exptXXYY.

        See equation E13 of https://arxiv.org/abs/2012.09238.

        Applies the two-qubit entangling gate
            exp(i * τ * X_A ⊗ X_B) * exp(i * τ * Y_A ⊗ Y_B),
        which corresponds to the interaction:
            F† * exp(i * 2τ * a2†a2) * exp(−i*2τ*a3†a3) · F

        This operator has matrix representation:
            [[1,       0,          0,        0],
            [0,   cos(2τ),   i sin(2τ),     0],
            [0,   i sin(2τ), cos(2τ),       0],
            [0,       0,          0,        1]]

        Args:
            batched_hamming_weight_phasing_qubrick (Qubrick): Qubrick for computing the hamming weight phasing.
                If None, rotations will be simulated directly.
            **kwargs: Other arguments to pass to the init.

        Note:
            - Note that we can applying rootx gate instead of HSH for Cliffords conjugating arbitrary angle rotations.
        """
        super().__init__(**kwargs)
        self.batched_hamming_weight_phasing_qubrick = batched_hamming_weight_phasing_qubrick
        if self.batched_hamming_weight_phasing_qubrick:
            self.n_hwp_batches = self.batched_hamming_weight_phasing_qubrick.n_hwp_batches
        else:
            self.n_hwp_batches = None

    def _compute(
        self,
        site_A_qubits: Qubits,
        site_B_qubits: Qubits,
        rz_rotation_angle: float,
        ctrl: int | Qubits = 0,
    ):
        """Computes exp(i*τ*X_A X_B)*exp(i*τ*Y_A Y_B) in parallel over qubit lists A and B. We use the gate decomposition here versus using the
           PPR representation of the XX and YY terms.

        Args:
            site_A_qubits: Firsts of pairs of qubits in plaquette.
            site_B_qubits: Seconds of pairs of qubits in plaquette.
            rz_rotation_angle: Rotation angle - depends on the evolution time and Hamiltonian
            ctrl: The quantum controls that control the action of the Qubrick.

        Note:
            - In the context of nearest-neighbour hopping plaquettes, site_A_qubits refer to qubits on sites two, and
              site_B_qubits refer to qubits on sites three.
        """
        if self.n_hwp_batches and (self.n_hwp_batches >= (site_A_qubits | site_B_qubits).num_qubits):
            raise ValueError("""Number of HWP batches must be less than
                the size of the target register. Try dividing by a power of 2.""")

        (site_A_qubits | site_B_qubits).had()
        (site_A_qubits | site_B_qubits).s()
        (site_A_qubits | site_B_qubits).had()
        for index in range(len(site_A_qubits)):
            site_B_qubits[index].x(site_A_qubits[index])

        site_A_qubits.had()

        target_qubits = site_A_qubits | site_B_qubits

        if self.batched_hamming_weight_phasing_qubrick:
            target_qubits = site_A_qubits | site_B_qubits
            self.batched_hamming_weight_phasing_qubrick.compute(rz_rotation_angle, target_qubits, ctrl=ctrl)
            self.catalyst = self.batched_hamming_weight_phasing_qubrick.get_result_qreg(
                "catalyst_reg"
            )  # This property is to anticipate potentially releasing this specific catalyst in the middle of the circuit versus the very end.
        else:
            (site_A_qubits | site_B_qubits).rz(rz_rotation_angle, cond=ctrl)

        site_A_qubits.had()

        for index in range(len(site_A_qubits)):
            site_B_qubits[index].x(site_A_qubits[index])
        (site_A_qubits | site_B_qubits).had()
        (site_A_qubits | site_B_qubits).s_inv()
        (site_A_qubits | site_B_qubits).had()


class exptXXYYViaPPR(Qubrick):
    """Implement the operator for exp(i*τ*X_A X_B)*exp(i*τ*Y_A Y_B) evolution
    using Pauli Product Rotation (PPR) decomposition.

    See equation E13 of https://arxiv.org/abs/2012.09238.

    Applies the two-qubit entangling gate
            exp(i * τ * X_A ⊗ X_B) · exp(i * τ * Y_A ⊗ Y_B),
                which corresponds to the interaction:
            F† · exp(i·2τ·a2†a2) · exp(-i·2τ·a3†a3) · F

        This operator has matrix representation:
            [[1,       0,          0,        0],
            [0,   cos(2τ),   i·sin(2τ),     0],
            [0,   i·sin(2τ), cos(2τ),       0],
            [0,       0,          0,        1]]
    """

    def __init__(
        self, batched_hamming_weight_phasing_qubrick: Qubrick | None = None, use_black_box: bool = False, **kwargs
    ):
        """Initialize qubrick implementing exptXXYY via PPR decomposition.

        Args:
            batched_hamming_weight_phasing_qubrick: Qubrick for computing the hamming weight phasing.
                If None, rotations will be simulated directly.
            use_black_box: If True, uses a black box implementation with optimized AV counters.
            **kwargs: Other arguments to pass to the init.

        Note:
            - Uses PPR decomposition for the Cliffords.
        """
        super().__init__(**kwargs)
        self.batched_hamming_weight_phasing_qubrick = batched_hamming_weight_phasing_qubrick
        if self.batched_hamming_weight_phasing_qubrick:
            self.n_hwp_batches = self.batched_hamming_weight_phasing_qubrick.n_hwp_batches
        else:
            self.n_hwp_batches = None
        self.use_black_box = use_black_box

    def _compute(
        self,
        site_A_qubits: Qubits,
        site_B_qubits: Qubits,
        rz_rotation_angle: float,
        ctrl: int | Qubits = 0,
    ):
        """Computes exptXXYY on qubit lists A and B. We use the PPR decomposition of the X_A X_B
           and Y_A Y_B terms.

        Args:
            site_A_qubits: First qubits of pairs of qubits in plaquette.
            site_B_qubits: Second qubits of pairs of qubits in plaquette.
            rz_rotation_angle: Rotation angle - depends on the evolution time and Hamiltonian
            ctrl: The quantum controls that control the action of the Qubrick.

        Note:
            - In the context of plaquettes, site_A_qubits refer to qubits on sites two, and
              site_B_qubits refer to qubits on sites three.
        """
        if self.n_hwp_batches and (self.n_hwp_batches >= (site_A_qubits | site_B_qubits).num_qubits):
            raise ValueError("""Number of HWP batches must be less than
                the size of the target register. Try dividing by a power of 2.""")

        for qbit in site_A_qubits:
            qbit.ppr(np.pi / 2 * units.rad, 1, 0)
        # XY pi/4 PPR
        if self.use_black_box:
            # Cost is 10: with an odd number of Ys in the PPR, using the
            # right-most identity avoids having to create an additional Y state.
            # See Fig. 11 of arXiv:1808.02892.
            num_parallel_applications = site_A_qubits.num_qubits
            av = 10 * num_parallel_applications  # Adding compute and uncompute cost of XY PPR angle pi/4
            cost = QubrickCosts(active_volume=av)
            self.get_qc().add_cost_event(cost)
        else:
            for qbit2, qbit3 in zip(site_A_qubits, site_B_qubits):
                (qbit2 | qbit3).ppr(-np.pi / 4 * units.rad, 3, 1)

        target_qubits = site_A_qubits | site_B_qubits

        if self.batched_hamming_weight_phasing_qubrick:
            target_qubits = site_A_qubits | site_B_qubits

            self.batched_hamming_weight_phasing_qubrick.compute(rz_rotation_angle, target_qubits, ctrl)
            self.catalyst = self.batched_hamming_weight_phasing_qubrick.get_result_qreg(
                "catalyst_reg"
            )  # This property is to anticipate potentially releasing this specific catalyst in the middle of the circuit versus the very end.
        else:
            (site_A_qubits | site_B_qubits).rz(rz_rotation_angle, cond=ctrl)

        if not self.use_black_box:
            for qbit2, qbit3 in zip(site_A_qubits, site_B_qubits):
                (qbit2 | qbit3).ppr(np.pi / 4 * units.rad, 3, 1)
        for qbit in site_A_qubits:
            qbit.ppr(-np.pi / 2 * units.rad, 1, 0)


class TwoModeFFFTViaControlledHad(Qubrick):
    """Implement the two mode FFFT transform via controlled Hadamard gates."""

    def __init__(self, **kwargs):
        """Implement a tower of :math:`F_{i,j}` operators in Fig. 8 of arXiv:1902.10673v4.
           The two mode FFFT transform to help diagonalize a plaquette operator.

        Args:
            **kwargs: Other arguments to pass to the init.

        Note:
            - This uses the decomposition shown in Fig. 5 of https://arxiv.org/abs/1912.06007.
        """
        super().__init__(**kwargs)

    def _compute(self, qubits_i: Qubits | list[Qubits], qubits_j: Qubits | list[Qubits], ctrl: int | Qubits = 0):
        """Apply the two mode FFFT transform.

        Args:
            qubits_i: The first qubit(s) to apply two-mode FFFT onto; can
                be a list of Qubits
            qubits_j: The second qubit(s) to apply two-mode FFFT onto; can
                be a list of Qubits
            ctrl: Control qubit

        Note:
            - Implements a stack of two-mode FFFTs on the inputs such that
              we can line up the circuit and more easily visualize, e.g.,
              the towers of rotations in the plaquette terms.
        """
        for index in range(len(qubits_i)):
            qubits_i[index].x(qubits_j[index])

        for index in range(len(qubits_i)):
            qubits_j[index].had(qubits_i[index] | ctrl)

        for index in range(len(qubits_i)):
            qubits_i[index].x(qubits_j[index])

        for index in range(len(qubits_i)):
            (qubits_i[index] | qubits_j[index] | ctrl).reflect()


class TwoModeFFFTViaPPRs(Qubrick):
    """Implement the two mode FFFT transform via Pauli product rotations (PPRs)."""

    def __init__(self, use_black_box: bool = False, **kwargs):
        """Implement a tower of the :math:`F_{i,j}` operators in Fig. 8 of arXiv:1902.10673v4.
           The two mode FFFT transform to help diagonalize a plaquette operator.

        Args:
            use_black_box: If True, uses a black box implementation with optimized AV counters.
            **kwargs: Other arguments to pass to the init.

        Note:
            - This uses the PPR decomposition of F. Still uses 2 T-type gates but this
            decomposition has a lower AV.

        """
        super().__init__(**kwargs)
        self.use_black_box = use_black_box

    def _compute(self, qubits_i: Qubits | list[Qubits], qubits_j: Qubits | list[Qubits], ctrl: int | Qubits = 0):
        """Apply the two mode FFFT transform using the PPR decomposition.

        Args:
            qubits_i: The first qubit(s) to apply F onto; can
                be a list of Qubits
            qubits_j: The second qubit(s) to apply F onto; can
                be a list of Qubits
            ctrl: Control qubit

        Note:
            - For this decomposition to exactly equal the desired unitary,
              add the following lines for applying a global phase at the end:

              for index in range(len(i)):
                  (j[index]|i[index]).ppr(np.pi/2, 0, 0)
            - This also means all the tests will have to be equivalent
              up to a global phase
            - Implements a stack of two-mode FFFTs on the inputs such that
              we can line up the circuit and more easily visualize, e.g.,
              the towers of rotations in the plaquette terms.
        """
        if self.use_black_box:
            # Note: This count does not describe the valid two mode FFFT transform unitary without the compute/uncompute pair.
            # We count the ZX-optimized block count for XX and YY pi/8 rotations
            # and one S-like op (pi/4-Z). This is specifically for when these Givens are inside a plaquette operator
            # so that the compute and uncompute appear together with no intermediate gates on one of the two qubits.
            # Originally, there are two of these S ops
            # conjugating the XX and YY PPRs, but the inner S ops from two-mode FFFT
            # and its dagger cancel in this specific case.
            n_parallel_ffft = len(qubits_i)
            av_xx_yy_pi8 = 64  # ZX-optimized value
            av_s = op_av_lookup_table[QPU_op(opcode=opc.OP_qc_s, target=1, condition=0)]
            av = (av_xx_yy_pi8 + av_s) * n_parallel_ffft
            cost = QubrickCosts(active_volume=av, t_gates=2 * n_parallel_ffft)

            self.get_qc().add_cost_event(cost)
        else:
            angle = np.pi / 8 * units.rad

            for index in range(len(qubits_i)):
                (qubits_j[index] | qubits_i[index]).ppr(-1 * angle, 3, 1, cond=ctrl)

            for index in range(len(qubits_i)):
                (qubits_j[index] | qubits_i[index]).ppr(angle, 3, 2, cond=ctrl)
            for index in range(len(qubits_i)):
                (qubits_j[index] | qubits_i[index]).ppr(np.pi / 2 * units.rad, 0, 1, cond=ctrl)
