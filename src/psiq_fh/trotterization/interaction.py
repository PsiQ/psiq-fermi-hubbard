"""Qubricks for implementing the interaction term in the Fermi-Hubbard Trotterizaton circuit in arxiv:2012.09238."""

import numpy as np
from psiqdk.workbench import Qubrick


class InteractionTrotterStep(Qubrick):
    """Implements time evolution of the Interaction terms in the Hubbard model - exp(itH_I) - based on 2012.09238."""

    def __init__(
        self,
        batched_hamming_weight_phasing_qubrick: Qubrick | None = None,
        **kwargs,
    ):
        """Construct the Trotter step for all Interaction terms of the Fermi-Hubbard Hamiltonian.

        Args:
            batched_hamming_weight_phasing_qubrick: Qubrick for computing the hamming weight phasing
                in batches for improved efficiency. If None, rotations will be simulated directly.
            **kwargs: Other arguments to pass to the init.
        """
        super().__init__(**kwargs)

        self._batched_hamming_weight_phasing_qubrick = batched_hamming_weight_phasing_qubrick
        if self.batched_hamming_weight_phasing_qubrick:
            self.n_hwp_batches = self.batched_hamming_weight_phasing_qubrick.n_hwp_batches
        else:
            self.n_hwp_batches = None

    @property
    def batched_hamming_weight_phasing_qubrick(self):
        """The batched hamming weight phasing qubrick.

        Returns:
            Qubrick: The batched hamming weight phasing qubrick used for computation
        """
        return self._batched_hamming_weight_phasing_qubrick

    def _compute(self, target_reg, data, ctrl=0):
        """Compute the time evolution term.

        Args:
            target_reg (Qubits): Qubit register storing the system. Register is of size number_of_spin_orbitals or
                2*x_dimension*y_dimension.
            data (InteractionTermData): Interaction term dataclass.
            ctrl (int or Qubits, optional): The quantum controls that control the action of the Qubrick.

        Note:
            - Unlike the hopping term, the rz rotation angle is re-computed in compute because
              evolution times are doubled/merged except for the last interaction term.
        """
        interaction_indices = data.interaction_indices
        self.evolution_time = data.evolution_time
        interaction_coefficient = data.coefficient
        particle_hole_symmetry = data.particle_hole_symmetry

        if self.n_hwp_batches and (self.n_hwp_batches >= target_reg.num_qubits):
            raise ValueError("""Number of HWP batches must be less than
                the size of the target register.""")

        qc = self.get_qc()

        # Sign: angle is negative because the sign of potential term is positive,
        # and the sign of Rz is opposite of sign of PPRs by gate definition.
        # Factor of 2: from Rz(theta) having diagonal elements
        # of exp(-i (theta/2)) and exp(i (theta/2)) where as PPR (including
        # for XX+YY) is defined without this factor of 2, as exp(i theta (XX+YY))
        angle = -1 * interaction_coefficient * self.evolution_time * 2 * 180 / np.pi

        rotation_reg = 0
        for term in interaction_indices:
            target_reg[term[1]].x(target_reg[term[0]])
            rotation_reg = rotation_reg | target_reg[term[1]]

        if self.batched_hamming_weight_phasing_qubrick:
            self.batched_hamming_weight_phasing_qubrick.compute(angle, rotation_reg, ctrl)
            self.catalyst1 = self.batched_hamming_weight_phasing_qubrick.get_result_qreg("catalyst_reg")
        else:
            qc.rz(angle, rotation_reg, ctrl)

        for term in interaction_indices:
            target_reg[term[1]].x(target_reg[term[0]])

        if not particle_hole_symmetry:
            if self.batched_hamming_weight_phasing_qubrick:
                self.batched_hamming_weight_phasing_qubrick.compute(-angle, target_reg, ctrl)
                self.catalyst2 = self.batched_hamming_weight_phasing_qubrick.get_result_qreg("catalyst_reg")
            else:
                qc.rz(-angle, target_reg, ctrl)
