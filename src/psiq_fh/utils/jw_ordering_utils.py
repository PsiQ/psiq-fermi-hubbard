"""Functions to generate and manipulate Jordan-Wigner ordering of the fermionic modes."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

### Functions to generate Jordan-Wigner ordering enumerations


def pink_happy_enum(L_x: int, L_y: int | None = None):
    """Generate a (L_x x L_y) NumPy array with values snaking right then left,
    forming 2x2 squares so that the pink plaquettes from arxiv:2012.09238 are local.

    Args:
        L_x (int): The size of the lattice in the x direction. Must be even.
        L_y (int): The size of the lattice in the y direction, if None assume square. Must be even.

    Returns:
        np.ndarray: A (L_x x L_y) matrix with the pink snaking pattern.

    Example:
        >>> pink_happy_enum(L_x=2, L_y=4)
            array([[0, 3, 4, 7],
                   [1, 2, 5, 6]])
    """
    if L_x % 2 != 0:
        raise ValueError("Lattice x dimension size must be even")
    if L_y is None:
        L_y = L_x
    if L_y % 2 != 0:
        raise ValueError("Lattice y dimension size must be even")

    pink_happy_enum_array = np.zeros((L_x, L_y), dtype=int)
    num = 0

    for row in range(0, L_x, 2):  # Process two rows at a time
        if (row // 2) % 2 == 0:
            # Snake right
            for col in range(L_y):
                if col % 2 == 0:
                    pink_happy_enum_array[row, col] = num
                    num += 1
                    pink_happy_enum_array[row + 1, col] = num
                    num += 1
                else:
                    pink_happy_enum_array[row + 1, col] = num
                    num += 1
                    pink_happy_enum_array[row, col] = num
                    num += 1
        else:
            # Snake left
            for col in range(L_y - 1, -1, -1):
                if col % 2 == 0:
                    pink_happy_enum_array[row + 1, col] = num
                    num += 1
                    pink_happy_enum_array[row, col] = num
                    num += 1
                else:
                    pink_happy_enum_array[row, col] = num
                    num += 1
                    pink_happy_enum_array[row + 1, col] = num
                    num += 1

    return pink_happy_enum_array


def generate_high_low_enum(L_x: int, L_y: int | None = None, pink_happy: bool = False):
    """Generate an enumeration pattern with spin-up assigned to low indices and spin-down to high indices.

    This layout is intended to make fSWAP operations more efficient, as the interaction term is already local.

    Args:
        L_x: The size of the lattice (must be even) in the x axis.
        L_y: The size of the lattice (must be even) in the y axis.
        pink_happy: If True, generates a snaking pattern so that pink plaquettes are local.
            If False, uses a simple left-to-right pattern.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two (lattice_size x lattice_size) arrays representing
            the spin-up and spin-down sector enumerations.
    """
    if L_x % 2 != 0:
        raise ValueError("Lattice x dimension size must be even")
    if L_y is None:
        # Assume square.
        L_y = L_x
    if L_y % 2 != 0:
        raise ValueError("Lattice y dimension size must be even")

    if pink_happy:
        spin_up_sector = pink_happy_enum(L_x, L_y=L_y)
    else:
        spin_up_sector = np.arange(L_x * L_y, dtype=int).reshape(L_x, L_y)
    spin_down_sector = spin_up_sector + L_x * L_y
    return (spin_up_sector, spin_down_sector)


def generate_even_odd_enum(L_x: int, L_y: int | None = None):
    """Generate an enumeration pattern with spin-up assigned to even indices and spin-down to odd indices.

    Args:
        L_x: The size of the lattice in the x axis, must be even.
        L_y: The size of the lattice in the y axis, must be even.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two (lattice_size x lattice_size) arrays representing
            the spin-up and spin-down sector enumerations.
    """
    if L_x % 2 != 0:
        raise ValueError("Lattice x dimension size must be even")
    if L_y is None:
        # Assume square.
        L_y = L_x
    if L_y % 2 != 0:
        raise ValueError("Lattice y dimension size must be even")

    spin_up_sector = np.arange(L_x * L_y, dtype=int).reshape(L_x, L_y) * 2
    spin_down_sector = spin_up_sector + 1
    return (spin_up_sector, spin_down_sector)


def validate_enumerations(enumerations: list[NDArray]) -> None:
    """Validate enumerations.

    Args:
        enumerations (List[np.darray]): list of enumerations.

    Raises:
        ValueError if any of the below criteria are not met.

    Note:
        - Checks that we only have maximum 2 enumerations as we only support spin 1/2 or spinless.
        - Checks that the enumerations have the same shape as we currently assume that the multiple
          enumerations correspond to different spin sectors and that every point in the lattice has the same spin.
        - Checks that all the indices are unique, both within the enumeration and between them.
    """
    # Check that there are max 2 enumerations.
    if not (1 <= len(enumerations) <= 2):
        raise ValueError("Currently only 1 or 2 spin sectors supported.")

    # Check that the shape of the enumerations are the same.
    reference_shape = enumerations[0].shape
    for i, arr in enumerate(enumerations[1:], start=1):
        if arr.shape != reference_shape:
            raise ValueError(
                f"All enumerations must have the same shape as correspond to different spin sectors. "
                f"Mismatch found at index {i}: expected {reference_shape}, got {arr.shape}"
            )

    # Check that
    if (x := sorted(set(col for enum in enumerations for row in enum for col in row))) != (
        y := sorted(index for enum in enumerations for index in enum.flatten())
    ):
        raise ValueError(f"Duplicated indices found in enumerations.\n\tUnique indices: {x}\n\tAll indices: {y}")


### Functions to manipulate orderings


def generate_interaction_enumeration_indices(enumerations: list[NDArray]) -> list[list[int, int]]:
    """Generate interaction indices from enumerations.

    Args:
        enumerations: list of numpy arrays, enumerations for the spin up and spin down sectors.

    Returns:
        List: interaction_indices, pink_plaquette_indices, gold_plaquettes_indices.

    Note:
        - If only one spin sector, interaction indices is an empty list.

    Example:
        For: spin_up = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
             spin_down = np.array([[8, 9, 10, 11], [12, 13, 14, 15]])
        Calling this function on [spin_up, spin_down] returns:
            [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]
    """
    validate_enumerations(enumerations)
    if not (1 <= len(enumerations) <= 2):
        raise ValueError("Currently only 1 or 2 spin sectors supported.")

    # Get interaction indices in the spinful case.
    interaction_indices = []
    if len(enumerations) == 2:
        interaction_indices = [
            [first_interaction_index, second_interaction_index]
            for first_interaction_index, second_interaction_index in zip(
                enumerations[0].flatten(), enumerations[1].flatten()
            )
        ]
    return interaction_indices


def generate_plaquette_enumeration_indices(
    enumerations: list[NDArray],
    color: Literal["pink", "gold"],
):
    """Generates indices from enumerations.

    Args:
        enumerations: list of numpy arrays, enumerations for the spin up and spin down sectors.
        color: describes whether the plaquette is either pink or gold (see arxiv:2012.09238).

    Returns:
        List: plaquette_indices (can be pink or gold type)

    Example:
        For: spin_up = np.array([[0,1,2,3],[4,5,6,7]])
             spin_down = np.array([[8,9,10,11],[12,13,14,15]])
        Calling this function on [spin_up,spin_down] with color="pink" returns:
            [[0, 4, 5, 1], [2, 6, 7, 3], [8, 12, 13, 9], [10, 14, 15, 11]]
        And calling this function on [spin_up,spin_down] with color="gold" returns:
            [[5, 1, 2, 6], [7, 3, 0, 4], [13, 9, 10, 14], [15, 11, 8, 12]]
    """
    validate_enumerations(enumerations)
    # Check a valid color was given
    if color not in {"pink", "gold"}:
        raise ValueError(f"Invalid color '{color}'; must be 'pink' or 'gold'.")

    if color == "gold":
        enumerations = [shift_enumeration(enumeration) for enumeration in enumerations]

    # First get the plaquettes as 2x2 numpy arrays
    plaquettes = get_plaqs(enumerations)

    # Convert this to lists of 4 element lists to fit with current implementations
    plaquette_indices = []
    for plaquette in plaquettes:
        plaquette_list = [plaquette[0][0], plaquette[1][0], plaquette[1][1], plaquette[0][1]]
        plaquette_indices.append(plaquette_list)

    # In the 2x2 case we don't consider the gold plaquette -- if we have periodic boundary conditions
    # the gold plaquette and pink plaquette are the same and so don't need to do Trotter. To apply
    # periodic boundary conditions in the 2x2 case this is equivalent to multiplying the kinetic coefficient
    # by 2
    if color == "gold" and len(plaquette_indices) == len(enumerations):
        plaquette_indices = []

    return plaquette_indices


def shift_enumeration(enumeration: NDArray):
    """Shift a square enumeration pattern so that pink indices become gold.
    This operates on a single spin sector represented by an n x n matrix.

    Args:
        enumeration (np.ndarray): A square (n x n) enumeration pattern.

    Returns:
        np.ndarray: The shifted (n x n) enumeration pattern.
    """
    rows, cols = enumeration.shape
    shifted_enumeration = np.zeros_like(enumeration)

    for i in range(rows):
        for j in range(cols):
            new_i = (i - 1) % rows  # Shift up with wrap-around
            new_j = (j - 1) % cols  # Shift left with wrap-around
            shifted_enumeration[new_i, new_j] = enumeration[i, j]

    return shifted_enumeration


def get_plaqs(
    enumerations: NDArray | list[NDArray], snaked: bool = False, plaquette_color: Literal["pink", "gold"] = "pink"
):
    """Returns an array of 2 x 2 plaquettes from one or more spin-sector enumerations.

    Accepts either:
        - a single 2-D array (L x L), or
        - a sequence (list/tuple) of 2-D arrays with identical shape, e.g. [up, down].

    Args:
        enumerations: np.ndarray (L x L) or sequence of such arrays.
        snaked: If True, reverse every other 2-row block to create a snaking order.
        plaquette_color: "pink" (default) or "gold". For "gold", each sector is shifted.

    Returns:
        np.ndarray with shape (S * N, 2, 2) where:
            S = number of spin sectors provided (1 or 2),
            N = number of plaquettes per sector = (L//2) * (L//2).
    """
    #  Normalize input to a list of 2-D arrays (one per spin sector)
    if isinstance(enumerations, np.ndarray):
        if enumerations.ndim != 2:
            raise ValueError(f"Expected 2-D enumeration (L x L); got shape {enumerations.shape}")
        enums = [enumerations]
    else:
        # Treat as a sequence of arrays
        try:
            enums = [np.asarray(e) for e in enumerations]
        except Exception:
            # Not actually iterable → treat as single array
            e = np.asarray(enumerations)
            if e.ndim != 2:
                raise ValueError(f"Expected 2-D enumeration (L×L); got shape {e.shape}")
            enums = [e]

    if len(enums) == 0:
        raise ValueError("enumerations must be a 2-D array or a non-empty sequence of 2-D arrays")

    # Validate shapes (all sectors must match)
    rows, cols = enums[0].shape
    if enums[0].ndim != 2:
        raise ValueError(f"enumeration must be 2-D (L×L); got shape {enums[0].shape}")
    for i, e in enumerate(enums[1:], start=1):
        if e.ndim != 2 or e.shape != (rows, cols):
            raise ValueError(
                f"All enumerations must be 2-D with the same shape; "
                f"got enum[{i}].shape={e.shape}, expected {(rows, cols)}"
            )

    # Shift for gold plaquettes per sector
    if plaquette_color == "gold":
        enums = [shift_enumeration(e) for e in enums]

    # Extract 2×2 plaquettes for each sector
    n_total_plaqs = (rows // 2) * (cols // 2)
    plaqs = np.empty((len(enums), n_total_plaqs, 2, 2), dtype=enums[0].dtype)

    for spin, enumeration in enumerate(enums):
        k = 0
        for i in range(0, rows, 2):
            col_indices = list(range(0, cols, 2))
            if snaked and ((i // 2) % 2 == 1):
                col_indices.reverse()
            for j in col_indices:
                plaqs[spin, k] = enumeration[i : i + 2, j : j + 2]
                k += 1

    # Flatten spin dimension so callers get (S*N, 2, 2)
    return plaqs.reshape(len(enums) * n_total_plaqs, 2, 2)
