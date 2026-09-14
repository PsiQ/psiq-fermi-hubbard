"""Module to test fswap networks (both ket and phase)."""

import random
from typing import Optional, Sequence

import numpy as np
import pytest
from psiqdk.workbench import QPU, Qubits

from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt
from psiq_fh.trotterization.fswapping.fswap_network import (
    NaivefSWAPNetworkWithReplace,
    PinkLocalizedFermionicSwapNetworkWithReplace,
)
from psiq_fh.utils.jw_ordering_utils import generate_high_low_enum, get_plaqs
from itertools import combinations
from typing import List


def get_elements_excluding_indices(arr, input_indices):
    """Return elements of an array where their indices are *not* in the input indices.

    Args:
        arr (np.array): input array
        input_indices (np.array): indices of input array to exclude

    Returns:
        list of original array without input indices
    """
    return [element for i, element in enumerate(arr) if i not in input_indices]


def find_consecutive_powers_sum(target_reg, count=4):
    """Find consecutive powers of two that sum to the given target value as
    read from a target register. Returns the powers if found, otherwise
    returns None.
    By default, this scans the register to find the first group of 4 consecutive bits
    that read as the integer 15 (i.e., "1111" in binary).

    Args:
        target_reg: A quantum register.
        count: The number of consecutive bits to sum to the target value.

    Returns:
        np.ndarray or None: A NumPy array of 4 consecutive indices whose
                            bits read as 15, or None if not found.
    """
    target_val = target_reg.read()
    # Total multiplier for powers of 2 sum: 2^n * (2^count - 1)
    total_factor = sum(2**i for i in range(count))  # e.g., 1 + 2 + 4 + 8 = 15 for count=4

    # Check if the target is divisible by total_factor
    if target_val % total_factor != 0:
        return None  # Not representable

    base_power_value = target_val // total_factor

    # Check if base_power_value is a power of 2
    if base_power_value & (base_power_value - 1) != 0:
        return None  # Not a power of two

    n = base_power_value.bit_length() - 1
    return np.array([(n + i) for i in range(count)])


@pytest.mark.parametrize("lattice_size", [4, 6])
@pytest.mark.parametrize("plaquette_color", ["pink", "gold"])
@pytest.mark.parametrize(
    "network_qubrick_cls",
    [NaivefSWAPNetworkWithReplace, PinkLocalizedFermionicSwapNetworkWithReplace],
)
@pytest.mark.parametrize("enumeration", ["high_low_pink_happy"])
def test_fswap_network_ket(lattice_size, plaquette_color, network_qubrick_cls, enumeration):
    """Clifford-sim test: each plaquette prepared as |1111⟩ should be localized to four consecutive qubits
    by the chosen fSWAP network, with all other qubits left at |0⟩.

    Notes:
        - For 'pink', we localize pink plaquettes from the given enumeration.
        - For 'gold', we localize gold plaquettes by using shifted enumerations for indexing.
        - We do not check global phase.
    """
    # Enumeration selection
    if enumeration == "high_low_pink_happy":
        current_enumeration = generate_high_low_enum(lattice_size, pink_happy=True)
    else:
        raise ValueError(f"Unknown enumeration: {enumeration}")

    # Plaquette indices to flip (based on requested color)
    np_plaqs = get_plaqs(current_enumeration, plaquette_color=plaquette_color)

    fswap_network_qubrick = network_qubrick_cls(FermionicSwapWithReplace())
    plaquette_indices_orig = [[a[0, 0], a[1, 0], a[1, 1], a[0, 1]] for a in list(np_plaqs)]

    number_spin_sites = 2 * (lattice_size * lattice_size)
    total_num_qubits = number_spin_sites

    # Loop through each plaquette pattern
    localized_indices = []

    for idx in np.arange(len(plaquette_indices_orig)):
        qc = QPU(num_qubits=total_num_qubits, filters=[">>clifford-qpu>>", ">>buffer>>"])

        target_reg = Qubits(number_spin_sites, "psi", qc)

        # Prepare |1111⟩ on the selected plaquette indices
        target_reg[np.array(plaquette_indices_orig[idx])].x()

        fswap_network_qubrick.compute(target_reg, current_enumeration, plaquette_color)

        # The four flipped qubits should now be consecutive wires (sum mask == 15)
        consecutive = find_consecutive_powers_sum(target_reg)
        localized_indices.append(consecutive)
        assert consecutive is not None, "Did not find four consecutive localized wires."

        # Everything else should be |0⟩
        rest = get_elements_excluding_indices(np.arange(number_spin_sites), consecutive)
        assert target_reg[rest].read() == 0

    # Ensure each qubit participates exactly once across all localized plaquettes
    assert np.array_equal(
        np.sort(np.concatenate(localized_indices)),
        np.arange(number_spin_sites),
    )


def count_inversions(perm: Sequence[int]) -> int:
    """Return the number of inversions in a sequence.
    An inversion is a pair (i, j) with i < j and perm[i] > perm[j].

    Complexity: O(n log n) using merge-count.
    """
    arr = list(perm)

    def merge_count(left: list[int], right: list[int]) -> tuple[list[int], int]:
        out, i, j, inv = [], 0, 0, 0
        nL, nR = len(left), len(right)
        while i < nL and j < nR:
            if left[i] <= right[j]:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
                inv += nL - i  # all remaining in left are > right[j]
        if i < nL:
            out.extend(left[i:])
        if j < nR:
            out.extend(right[j:])
        return out, inv

    def sort_count(a: list[int]) -> tuple[list[int], int]:
        n = len(a)
        if n <= 1:
            return a, 0
        mid = n // 2
        L, invL = sort_count(a[:mid])
        R, invR = sort_count(a[mid:])
        merged, invM = merge_count(L, R)
        return merged, invL + invR + invM

    _, inv = sort_count(arr)
    return inv


def get_phase(initial_list: List[int], states: List[int], final_permutation: List[int]) -> int:
    """Return phase/sign from applying a fermionic swap block characterized
       by input and output lists of qubit label/enumeration as well as the states of each
       input qubit

    Args:
        initial_list: List of input qubit indices
        states: List of input computational basis state of the input qubit
            indices (1's and 0's)
        final_permutation: List of output qubit indices

    Returns:
        phase: Phase from the fSWAP block applied to an input computational
            basis state, -1 or 1.
    """
    # Step 1: Extract "on" elements from the initial list
    on_elements_initial = [x for x, s in zip(initial_list, states) if s == 1]

    # Step 2: Extract "on" elements in the final permutation, respecting state
    on_elements_final = [x for x in final_permutation if states[initial_list.index(x)] == 1]

    # Step 3: Build permutation indices
    perm_indices = [on_elements_initial.index(x) for x in on_elements_final]

    # Step 4: Count inversions to get parity
    inversions = count_inversions(perm_indices)
    phase = (-1) ** inversions
    return phase


def binary_string_to_list(binary_string):
    """Return list representation of a binary string."""
    if binary_string.startswith("0b"):
        binary_string = binary_string[2:]
    return [int(bit) for bit in binary_string]


def binary_list_to_int(binary_list):
    """Return integer representation of binary list."""
    binary_string = "".join(str(bit) for bit in binary_list)
    return int(binary_string, 2)


@pytest.mark.parametrize("fswap_qbk", [FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt])
def test_fswap_block_phase(fswap_qbk):
    """Test that the fswap block returns the expected phase/sign.

    Note:
        - We don't test the entire fswap network here because
          our clifford simulator does not keep track of global phases.
    """
    size_of_fswap_block = 6  # fix size

    # Try every pair of qubits to fswap over
    for i, j in list(combinations(np.arange(size_of_fswap_block), 2)):
        # Try every initial computational basis state
        for init_val in np.arange(2**size_of_fswap_block):
            qc = QPU(num_qubits=size_of_fswap_block)

            qubs = Qubits(size_of_fswap_block, "qubs", qc)
            qubs.write(init_val)

            fermionic_swap = fswap_qbk()
            fermionic_swap.compute(qubs, [i], [j])

            # get final qubit indices
            qubit_indices = np.arange(size_of_fswap_block)
            qubit_indices[i], qubit_indices[j] = qubit_indices[j], qubit_indices[i]

            # initial qubit indices
            initial = np.arange(size_of_fswap_block).tolist()  # LSB to MSB

            # convert initial computational state (integer representation)
            # into a list of 0's and 1's (padded to the qubit count)
            states = binary_string_to_list(bin(init_val))[::-1]
            if len(states) != len(initial):
                pad_size = len(initial) - len(states)
                states += [0] * pad_size
            states = np.array(states).tolist()
            final_indices = qubit_indices
            phase = get_phase(initial, states, final_indices)

            wf_dict = qc.pull_state(with_qreg_labels=True)["amps"]

            # Output is a computational basis state
            assert len(list(wf_dict.values())) == 1
            amp = list(wf_dict.values())[0]  # should be +1 or -1

            # Check phase
            assert amp == phase

            # Check ket
            # Permute to final value (output computational basis state)
            final_val_list = np.array(states)[np.array(final_indices)]
            final_val = binary_list_to_int(final_val_list[::-1])
            assert final_val == qc.read()


def sample_basis_states(
    n_qubits: int,
    k: int,
    *,
    seed: Optional[int] = None,
    replace: bool = False,
) -> list[int]:
    """Sample k computational basis states in [0, 2**n_qubits).

    Uses Python's big-integer RNG so it works for arbitrarily large n_qubits.

    Args:
        n_qubits: number of qubits.
        k: how many states to sample.
        seed: seed for determinism.
        replace: allow duplicates if True.

    Returns:
        List of integers in [0, 2**n_qubits).
    """
    rng = random.Random(seed)
    if replace:
        return [rng.getrandbits(n_qubits) for _ in range(k)]
    picked: set[int] = set()
    while len(picked) < k:
        picked.add(rng.getrandbits(n_qubits))
    return list(picked)


@pytest.mark.parametrize("lattice_size", [4, 6, 8])
@pytest.mark.parametrize("plaquette_color", ["gold", "pink"])
@pytest.mark.parametrize(
    "network_qubrick_cls",
    [NaivefSWAPNetworkWithReplace, PinkLocalizedFermionicSwapNetworkWithReplace],
)
@pytest.mark.parametrize("enumeration", ["pink_happy"])
def test_networks_agree_on_random_basis_states(
    lattice_size: int,
    plaquette_color: str,
    network_qubrick_cls: type,
    enumeration: str,
):
    n_site = lattice_size * lattice_size
    total_num_qubits = 2 * n_site

    # Deterministic sampling per param-set
    seed = hash((lattice_size, plaquette_color, network_qubrick_cls.__name__, enumeration)) & 0xFFFFFFFF
    selected_basis_states = sample_basis_states(total_num_qubits, k=6, seed=seed, replace=False)

    # Enumeration
    if enumeration == "pink_happy":
        current_enumeration = generate_high_low_enum(lattice_size, pink_happy=True)
    else:
        raise ValueError(f"Unknown enumeration: {enumeration!r}")

    base_net = network_qubrick_cls(FermionicSwapWithReplace())
    opt_net = network_qubrick_cls(FermionicSwapWithReplaceAVOpt())

    # Run comparisons on each sampled basis state
    for init_val in selected_basis_states:
        #  Base
        qc_base = QPU(filters=[">>buffer>>", ">>clifford-sim>>"])
        qc_base.reset(total_num_qubits)
        reg = Qubits(total_num_qubits, "psi_base", qc_base)
        reg.write(init_val)

        base_net.compute(reg, current_enumeration, plaquette_color)

        #  Optimized
        qc_opt = QPU(filters=[">>buffer>>", ">>clifford-sim>>"])
        qc_opt.reset(total_num_qubits)
        reg = Qubits(total_num_qubits, "psi_opt", qc_opt)
        reg.write(init_val)

        opt_net.compute(reg, current_enumeration, plaquette_color)

        assert qc_base.pull_state() == qc_opt.pull_state()  # Output of pull_state for clifford sim is a string
