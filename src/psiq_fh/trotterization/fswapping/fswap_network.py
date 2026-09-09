"""This file contains the qubrick implementations for fswap networks."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from psiqdk.workbench import Qubrick

from psiq_fh.trotterization.fswapping.fermionic_swap import (
    FermionicSwapWithReplace,
    FermionicSwapWithReplaceAVOpt,
)
from psiq_fh.utils.jw_ordering_utils import (
    generate_high_low_enum,
    generate_plaquette_enumeration_indices,
    get_plaqs,
    shift_enumeration,
)


def _as_class_or_instance_ok(obj, base) -> bool:
    """Check for class or instance of a class."""
    # If a class was passed, use issubclass; if an instance, use isinstance.
    if inspect.isclass(obj):
        return issubclass(obj, base)
    return isinstance(obj, base)


def get_fswap_with_replace_list(current_enumeration, target_enumeration, trivial_swaps=False, printing=False):
    """Generate a list of swaps to transform the current enumeration into the target enumeration.

    Args:
        current_enumeration (np.ndarray): An array of shape (n, 2, 2) containing plaquettes in their
            original numbering.
        target_enumeration (np.ndarray): An array of shape (n, 2, 2) containing plaquettes in the desired
            (local) numbering.
        trivial_swaps (bool): If True, include trivial swaps (i.e., swapping an element with itself)
            in the returned list.
        printing (bool): If True, print debug information during execution.

    Returns:
        np.ndarray: An array of shape (n', 2) with integer dtype, where each row specifies a pair of
            indices to be swapped to achieve the target enumeration.

    """
    if current_enumeration.shape != target_enumeration.shape:
        print("current_enumeration.shape", current_enumeration.shape)
        print("target_enumeration.shape", target_enumeration.shape)
        raise ValueError("Target and current enumerators must be the same shape")

    list_of_swaps = []
    for i in range(target_enumeration.shape[0]):
        if printing:
            print("Current plaquette \n", current_enumeration[i], "\n target plaquette \n", target_enumeration[i], "\n")
        # Currently swap to the orientation of the local plaquette in the target enumeration, potential to optimize second step. Optimisation 2 i.e. given the 4 local indices there are 8 local configurations (4 rotations and both clockwise and anticlockwise orientation) optimize over these for the minimum cost of the combined swaps.
        for k in range(2):
            for l_index in range(2):
                # This ordering does the swaps in the order [top left, bottom left, bottom right, top right]
                current = current_enumeration[i][l_index ^ k][k]
                target = target_enumeration[i][l_index ^ k][k]
                if current != target or trivial_swaps:
                    # Perform fSWAP between current and target
                    list_of_swaps.append([current, target])
                    # Update enumeration
                    target_location = np.where(current_enumeration == target)
                    current_enumeration[i][l_index ^ k][k] = target
                    current_enumeration[target_location] = current
    swap_array = np.asarray(list_of_swaps)
    return swap_array


class NaivefSWAPNetworkWithReplace(Qubrick):
    """Implement the naive fSWAP network with replace to localize the plaquettes in order for the plaquette evolution
    as described in https://arxiv.org/pdf/2012.09238 to take place in parallel in the circuit.
    This naive implementation does an unoptimized fSWAP routine, both the number and size of the fSWAP is subject
    further optimisation.

    fSWAP-With-Replace swaps qubits `a` and `b`(with `a < b`) while
    restoring all intermediate qubits to their original order.

    Note: While this is unoptimized it will work for any enumeration.
    """

    def __init__(self, fermionic_swap_qubrick: Qubrick = None, **kwargs):
        """Initialize naive fSWAP network qubrick.

        Args:
            fermionic_swap_qubrick (Qubrick): qubrick that implements the non-local fermionic swap.
            **kwargs: Other arguments to pass to the init.
        """
        if fermionic_swap_qubrick is None:
            self.fermionic_swap_qubrick = FermionicSwapWithReplace()
        else:
            self.fermionic_swap_qubrick = fermionic_swap_qubrick

        if not _as_class_or_instance_ok(
            self.fermionic_swap_qubrick, (FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt)
        ):
            raise TypeError("This qubrick only accepts FermionicSwapWithReplace or FermionicSwapWithReplaceAVOpt.")

        super().__init__(**kwargs)

    def _compute(
        self,
        target_reg,
        current_enumeration,
        color_to_localize: Literal["pink", "gold"],
    ):
        """Naively implements the fSWAP network to localize the plaquettes.
        This qubrick takes in the enumeration and generates the appropriate colored plaquette
        indices in a default list order. It then brute force localizes by assuming the first
        plaquette should be [0,1,2,3] etc.

        For example, if the first plaquette in the list is `[0, 3, 1, 2]` (assume
        there is only one plaquette in this example), this qubrick applies a network of fSWAP blocks
        to go from: `[0, 3, 1, 2]` to `[0, 1, 2, 3]`.

        Args:
            target_reg (Qubits): Register holding system qubits to apply the localising fSWAPs to.
            current_enumeration (List[np.darray]): Current enumeration of indices on the lattice; it is a list to account for possible multiple spin sectors
                                 and each np.darray is then an array of the indices. `(up_enum, dn_enum)` raw L×L inputs (pink-localized).
            color_to_localize: Must be either 'pink' or 'gold', corresponding to the plaquette arrangement in https://arxiv.org/abs/2012.09238.
            d.a
        """
        plaquette_indices = generate_plaquette_enumeration_indices(current_enumeration, color_to_localize)
        localized_plaquettes = []
        i = 0
        for plaquette in plaquette_indices:
            expected_plaquette_indices = [
                i,
                i + 1,
                i + 2,
                i + 3,
            ]  # Localized plaquette indices
            for index in range(4):
                current_index = plaquette[index]
                expected_index = expected_plaquette_indices[index]
                self.fermionic_swap_qubrick.compute(
                    target_reg, [current_index], [expected_index]
                )  # Perform the nonlocal swap
                # Update fermionic labels to adhere to current reordering
                for _plaquette in plaquette_indices:
                    for _index in range(4):
                        if _plaquette[_index] == expected_index:
                            _plaquette[_index] = current_index
                plaquette[index] = expected_index

            localized_plaquettes.append(
                expected_plaquette_indices
            )  # Store plaquette indices for performing localized plaquette evolutions
            i += 4
        self.set_classical_result(localized_plaquettes, "localized_plaquettes")


class PinkLocalizedFermionicSwapNetworkWithReplace(Qubrick):
    """Assuming a pink localized enumeration, implement an improved fSWAP network with replace to localize the plaquettes in order for the plaquette evolution
    as described in https://arxiv.org/pdf/2012.09238 to execute in parallel in the circuit.
    "pink localized" enumeration refers to an enumeration where the pink plaquettes are already localized (thus also called "pink happy"), and so the only fswaps that need to be executed are for the gold plaquettes.

    fSWAP-With-Replace swaps qubits `a` and `b`(with `a < b`) while
    restoring all intermediate qubits to their original order.

    Note: This is specifically designed for a pink happy enumeration, it will execute a correct circuit for other enumerations but will throw a warning.
    """

    def __init__(self, fermionic_swap_qubrick: Qubrick = None, **kwargs):
        """Initialize pink localized fSWAP network qubrick.

        Args:
            fermionic_swap_qubrick (Qubrick): qubrick that implements the non-local fermionic swap.
            **kwargs: Other arguments to pass to the init.
        """
        if fermionic_swap_qubrick is None:
            self.fermionic_swap_qubrick = FermionicSwapWithReplace()
        else:
            self.fermionic_swap_qubrick = fermionic_swap_qubrick

        if not _as_class_or_instance_ok(
            self.fermionic_swap_qubrick, (FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt)
        ):
            raise TypeError("This qubrick only accepts FermionicSwapWithReplace or FermionicSwapWithReplaceAVOpt.")

        super().__init__(**kwargs)

    def _compute(
        self,
        target_reg,
        current_enumeration,
        color_to_localize: Literal["pink", "gold"],
    ):
        """Implements the fSWAP network to localize the plaquettes, works generally but is designed to work with an already localized pink enumeration.
        This qubrick takes in the enumeration and generates the appropriate colored plaquette
        indices in a default list order. It then localizes by somewhat intuitively but further optimisation could be done.

        Args:
            target_reg (Qubits): Register holding system qubits to apply the localising fSWAPs to.
            current_enumeration (List[np.darray]): Current enumeration of indices on the lattice, is a list to account for possible multiple spin sectors and each np.darray is then a array of the indices.
            color_to_localize: Must be either 'pink' or 'gold', corresponding to the plaquette arrangement in https://arxiv.org/abs/2012.09238.
        """
        # Assuming the pink plaquettes are localized initially: if the color is pink all swaps should be trivial in the pink case and so nothing should happen
        # This is currently not hardcoded so that this qubrick can be called more generally.

        up_enumeration = current_enumeration[0]
        down_enumeration = current_enumeration[1]
        lattice_size = up_enumeration.shape[0]
        if color_to_localize == "pink":
            current_plaqs = get_plaqs([up_enumeration, down_enumeration], snaked=True)
        else:
            current_plaqs = get_plaqs(
                [shift_enumeration(up_enumeration), shift_enumeration(down_enumeration)], snaked=True
            )

        # target enumeration is the same as pink happy
        pink_happy_up, pink_happy_down = generate_high_low_enum(lattice_size, pink_happy=True)
        target_plaqs = get_plaqs([pink_happy_up, pink_happy_down], snaked=True)

        fswap_list = get_fswap_with_replace_list(current_plaqs, target_plaqs, trivial_swaps=False, printing=False)

        if color_to_localize == "pink" and len(fswap_list) > 0:
            # fswap list is not empty and the plaquette is pink. Which means the enumeration was not initially localized - raise a warning.
            warnings.warn(
                "Pink localized fswap network is being used without the pink localized initial enumeration.",
                UserWarning,
            )

        for fswap in fswap_list:
            current_index = fswap[0]
            expected_index = fswap[1]
            self.fermionic_swap_qubrick.compute(
                target_reg, [current_index], [expected_index]
            )  # Perform the nonlocal swap

        # Flatten the plaquettes to feed back to the plaquette evolution
        localized_plaquettes = [
            [target_plaq[0][0], target_plaq[1][0], target_plaq[1][1], target_plaq[0][1]] for target_plaq in target_plaqs
        ]
        self.set_classical_result(localized_plaquettes, "localized_plaquettes")
