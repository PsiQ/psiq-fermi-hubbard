"""Module for fermionic swap qubricks."""

import numpy as np
import psiqdk.workbench.opcodes._opcode_constants as opc
from psiqdk.workbench import Qubrick
from psiqdk.workbench.experimental._active_volume_estimation._av_counting._get_av_from_op import get_av_from_op
from psiqdk.workbench.experimental._active_volume_estimation._av_counting._qpu_op_functions import (
    _get_multi_target_cz_av,
)
from psiqdk.workbench.experimental.symbolics import QubrickCosts
from psiqdk.workbench.ops import QPU_op


def local_fswap_op(qc, qubit_i, qubit_j, ctrl=0):
    """Apply one nearest-neighbor fermionic swap between two adjacent wires.

        1) Apply the parity/sign operation required for fermions (e.g., a CZ on the
            parity mask covering the two targets and the control).
        2) Perform a (controlled) SWAP of the two target qubits.

    Args:
        qc:
            The qubit register to operate on.
        qubit_i:
            First (left) adjacent qubit endpoint.
        qubit_j:
            Second (right) adjacent qubit endpoint.
        ctrl:
            Control qubit.
    """
    # Parity/sign handling for fermions in this framework.
    qc.z(0, (qubit_i | qubit_j | ctrl))

    # Adjacent (controlled) SWAP between the two wires.
    qubit_i.swap(qubit_j, ctrl)


class FermionicSwapWithReplace(Qubrick):
    """Implement a fermionic swap with replace operation between qubits.

    fSWAP-With-Replace swaps qubits `a` and `b`(with `a < b`) while
    restoring all intermediate qubits to their original order.
    """

    def __init__(self, **kwargs):
        """Implement a fermionic swap between nonlocal fermionic sites.

        Args:
            **kwargs: Other arguments to pass to the init.
        """
        super().__init__(**kwargs)

    def _compute(
        self,
        qubits,
        indices_of_first_qubits,
        indices_of_second_qubits,
        ctrl=0,
    ):
        """Perform fermionic swaps between two sets of fermionic sites (potentially non-local).

        Note:
            We use lists of indices so that we can perform these operations "in parallel" if needed which makes the
                circuit drawings easier to see.

        Args:
            qubits (Qubits): The register holding the qubits contained within the swap
            indices_of_first_qubits (List[int]): A list of indices of the first qubits to "fermionically" swap
            indices_of_second_qubits (List[int]): A list of indices of the second qubit to "fermionically" swap
            ctrl (Qubit): Control qubit
        """
        assert len(indices_of_first_qubits) == len(indices_of_second_qubits)
        qc = self.get_qc()

        for a, b in zip(indices_of_first_qubits, indices_of_second_qubits):
            (a, b) = (min(a, b), max(a, b))

            if a == b:
                pass
            elif (a + 1) == b:
                local_fswap_op(qc, qubits[a], qubits[b], ctrl=ctrl)
            else:
                for index in range(b, a, -1):
                    local_fswap_op(qc, qubits[index], qubits[index - 1], ctrl=ctrl)

                for index in range(a + 1, b):
                    local_fswap_op(qc, qubits[index], qubits[index + 1], ctrl=ctrl)


class FermionicSwapWithReplaceAVOpt(Qubrick):
    """Optimized AV implementation of fermionic swap with replace.

    fSWAP-With-Replace swaps labels `a` and `b`(with `a < b`) while
    restoring all intermediate labels to their original order.
    """

    def __init__(self, use_black_box=False, **kwargs):
        """Implement a fermionic swap between nonlocal fermionic sites with comp.

        Args:
            use_black_box: If True uses a black box implementation with
                optimised custom AV counters, otherwise implements circuit
            **kwargs: Other arguments to pass to the init.
        """
        super().__init__(**kwargs)
        self.use_black_box = use_black_box

    def _compute(self, qubits, indices_of_first_qubits, indices_of_second_qubits, ctrl=0):
        """Perform fermionic swaps between two sets of fermionic sites (potentially non-local).

        Note:
            We use lists of indices so that we can perform these operations "in parallel" if needed which makes the
                circuit drawings easier to see.

        Args:
            qubits (Qubits): The register holding the qubits contained within the swap
            indices_of_first_qubits (List[int]): A list of indices of the first qubits to "fermionically" swap
            indices_of_second_qubits (List[int]): A list of indices of the second qubit to "fermionically" swap
            ctrl (Qubit): Control qubit
        """
        assert len(indices_of_first_qubits) == len(indices_of_second_qubits)
        qc = self.get_qc()

        for a, b in zip(indices_of_first_qubits, indices_of_second_qubits):
            (a, b) = (min(a, b), max(a, b))

            if a == b:
                pass
            elif (a + 1) == b:
                local_fswap_op(qc, qubits[a], qubits[b], ctrl=ctrl)
            else:
                c = np.arange(a + 1, b, dtype=int)
                if int(np.abs(a - b)) <= calc_crossover():
                    # Compilation 1
                    if self.use_black_box:
                        cost = QubrickCosts(active_volume=av_decomp1_fswap(abs(a - b)))
                        self.get_qc().add_cost_event(cost)
                    else:
                        qubits[a].swap(qubits[b])
                        (qubits[c] | qubits[a]).z(qubits[b])
                        qubits[c].z(qubits[a])
                else:
                    # Compilation 2
                    if self.use_black_box:
                        cost = QubrickCosts(active_volume=av_decomp2_fswap(abs(a - b)))
                        self.get_qc().add_cost_event(cost)
                    else:
                        qubits[b].swap(qubits[a])
                        qubits[a].z(qubits[b])
                        qubits[a].x(qubits[b])
                        qubits[c].z(qubits[a])
                        qubits[a].x(qubits[b])


def av_naive_decomp(n):
    """Function returns the AV count for the fSWAP of size n compiled naively, with replace.

    Args:
        n (int): size of fSWAP block, given we are trying to fswap modes i and i+n
    Returns:
        av_naive (int): number of active volume blocks
    """
    cz_op = QPU_op(opcode=opc.OP_qc_z, target=2, condition=1)
    cz_av = get_av_from_op(cz_op)
    return (2 * n - 1) * cz_av


def av_decomp2_fswap(n):
    """Function returns the by hand calculated AV count for the fSWAP of size n decomposition 2.
        Analytically the AV for this decomposition is ceil(3(n-1)/2) + 11.

    Args:
        n (int): size of fSWAP block, given we are trying to fswap modes i and i+n
    Returns:
        av_multiicz (int): number of active volume blocks
    """
    total_av = 0
    # First add the av of a CX and CZ (OZX-optimized)
    total_av += 5
    # Then add the av of a (n-1) target cZ
    total_av += _get_multi_target_cz_av(n - 1)
    # Then add the av of a CX (4 blocks in original AV paper)
    cx_op = QPU_op(opcode=opc.OP_qc_x, target=2, condition=1)
    cx_av = get_av_from_op(cx_op)
    total_av += cx_av
    return total_av


def av_decomp1_fswap(n):
    """Function returns the by hand calculated AV count for the fSWAP of size n decomposition 1.
        Analytically the AV for this decomposition is ceil(3n/2) + ceil(3(n-1)/2) + 4.

    Args:
        n (int): size of fSWAP block, given we are trying to fswap modes i and i+n
    Returns:
        av_multiicz (int): number of active volume blocks
    """
    total_av = 0
    # First add the av of a n-target CZ
    total_av += _get_multi_target_cz_av(n)
    # Then add the av of a (n-1) target cZ
    total_av += _get_multi_target_cz_av(n - 1)
    return total_av


def calc_crossover():
    """Function that calculates the crossover point for which of the two decomposition is favorable.

    n is the "size" of the fswap, i.e. when swapping fermion labelled i with fermion labelled i+n so in total n+1 qubits are involved in the swap.
    Both decompositions use a single control (m)-target CZ which costs CZ_AV(m) = int(np.ceil(3*m/2)) +2.
    The first decomposition has AV1(n) = ceil(3n/2) + ceil(3(n-1)/2) + 4.
    The second decomposition has AV2(n) = ceil(3(n-1)/2) + 11.
    The crosseover occurs at the n where ceil(3n/2) > 7.

    Returns:
        crossover (int): critical value of n after which one should use the second decomposition.
    """
    crossover = 0
    for n in range(1, 10):  # Check within a reasonable range
        # rearranged expression for AV_decomp1 - AV_decomp2 set to greater than 0
        if (np.ceil(3 * n / 2)) > 7:
            crossover = int(n)
            break
    return crossover
