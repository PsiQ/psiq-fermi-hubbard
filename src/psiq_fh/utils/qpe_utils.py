"""Module for computing number of queries for QPE."""

import numpy as np


def _compute_queries_sin(QPE_error: float, evolution_time: float, dpkb: bool = True, power_of_two: bool = True):
    """Query count for the two-ancilla sin-window QPE variant.

    Args:
        QPE_error: Target root-mean-squared error in the phase.
        evolution_time: Evolution time used in the QPE unitary.
        dpkb: If True, uses double phase kickback to approximately half the number of calls.
        power_of_two: Restrict the phase register to a power of two.

    Returns:
        int: Number of queries.
    """
    no_queries_sin_spkb = np.pi / np.arctan(QPE_error * evolution_time) - 1
    if not dpkb:
        if power_of_two:
            no_queries_sin = 2 ** np.ceil(np.log2(no_queries_sin_spkb))
        else:
            no_queries_sin = no_queries_sin_spkb
    else:
        # Solve for K: no_queries_sin_spkb = 2**K - 1
        K = np.log2(no_queries_sin_spkb + 1)
        # 2**(K-1) is the query count for uncontrolled U under dpkb. The first call to U is
        # actually controlled, but that cost is ignored here.
        if power_of_two:
            no_queries_sin = 2 ** (np.ceil(K) - 1)
        else:
            no_queries_sin = 2 ** (K - 1)

    return (np.ceil(no_queries_sin)).astype(int)


def _compute_queries_berry(QPE_error: float, evolution_time: float, dpkb: bool = True, power_of_two: bool = True):
    """Query count for the Berry et al. QPE variant.

    Args:
        QPE_error: Target root-mean-squared error in the phase.
        evolution_time: Evolution time used in the QPE unitary.
        dpkb: If True, uses double phase kickback to approximately half the number of calls.
        power_of_two: Restrict the phase register to a power of two.

    Returns:
        int: Number of queries.
    """
    M = 6
    no_queries_berry_spkb = (1.56 * np.pi) / (QPE_error * evolution_time)  # assumes M = 6
    if not dpkb:
        if power_of_two:
            no_queries_berry = 2 ** np.ceil(np.log2(no_queries_berry_spkb))
        else:
            no_queries_berry = no_queries_berry_spkb
    else:
        # Solve for K: no_queries_berry_spkb = M * (2**K - 1)
        K = np.log2(no_queries_berry_spkb / M + 1)
        if power_of_two:
            no_queries_berry = M * 2 ** (np.ceil(K) - 1)
        else:
            no_queries_berry = M * 2 ** (K - 1)

    return (np.ceil(no_queries_berry)).astype(int)


def compute_n_queries(
    QPE_error: float, evolution_time: float, sin_window: bool = False, dpkb: bool = True, power_of_two: bool = True
) -> int:
    """Number of phase estimation queries needed to reach a target RMS phase error.

    Counts calls to directionally-controlled unitaries e^{-i H t} needed to reach an RMS
    phase error of ``QPE_error``, as derived in https://arxiv.org/abs/1902.10673.

    Args:
        QPE_error: Target root-mean-squared error in the phase.
        evolution_time: Evolution time used in the QPE unitary.
        sin_window: Use the two-ancilla sin-window QPE variant. See Sect 2B of
            https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.041015.
        dpkb: If True, uses double phase kickback to approximately half the number of calls to the
            unitary by doing bidirectional control.
        power_of_two: Restrict the phase register to a power of two, so the final QFT acts on
            a power-of-two register.

    Returns:
        int: Estimated number of QPE oracle queries required.
    """
    if sin_window:
        return _compute_queries_sin(QPE_error, evolution_time, dpkb, power_of_two)

    return _compute_queries_berry(QPE_error, evolution_time, dpkb, power_of_two)
