"""This file contains the qubrick implementations for fswap networks."""

from __future__ import annotations

import inspect
import warnings

from typing import Literal

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
        # Swap to the orientation of the local plaquette in the target enumeration.
        # Note: potential to optimize second step - given 4 local indices there are 8 local configurations including clockwise and anticlockwise orientation
        # that we could optimize over to minimize cost of combined swaps.
        for k in range(2):
            for l_index in range(2):
                # This ordering does the swaps in the order [top left, bottom left, bottom right, top right] i.e. anticlockwise
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
    """Implement the naive fSWAP network with replace to localize the plaquettes in parallel.

    As described in https://arxiv.org/abs/2012.09238.

    Note:
        - This naive implementation performs an unoptimized fSWAP routine
        - While both the number and size of the fSWAP is subject to further optimization, this Qubrick will work for any enumeration.
        - If `FermionicSwapWithReplace` is used, it will swap qubits `a` and `b`(with `a < b`) while
          restoring all intermediate qubits to their original order.
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
        """Naive implementation of the fSWAP network to localize the plaquettes.

        This qubrick takes in an enumeration and generates the appropriate colored plaquette
        indices in a default list order. Assumes the first plaquette should be [0,1,2,3] etc.

        Args:
            target_reg (Qubits): Register holding system qubits to apply the fSWAPs to.
            current_enumeration (List[np.darray]): Current enumeration of indices on the lattice.
                                Note: it is a list to account for possible multiple spin sectors, e.g., pink-localized has `(up_enum, dn_enum)` where each has `L×L` inputs.
            color_to_localize: Must be either 'pink' or 'gold', corresponding to the plaquette arrangement as described in https://arxiv.org/abs/2012.09238.
        """
        plaquette_indices = np.array(generate_plaquette_enumeration_indices(current_enumeration, color_to_localize))

        localized_plaquettes = []
        i = 0
        for plaquette in plaquette_indices:
            # Localized plaquette indices
            expected_plaquette_indices = [i + j for j in range(4)]
            for index in range(4):
                current_index = plaquette[index]
                expected_index = expected_plaquette_indices[index]
                # Perform the nonlocal swap
                self.fermionic_swap_qubrick.compute(target_reg, [current_index], [expected_index])
                # Update plaquette to reflect this reordering
                plaquette_indices[plaquette_indices == expected_index] = current_index
                plaquette[index] = expected_index

            # Store plaquette indices for performing localized plaquette evolutions
            localized_plaquettes.append(expected_plaquette_indices)
            i += 4
        self.set_classical_result(localized_plaquettes, "localized_plaquettes")


class PinkLocalizedFermionicSwapNetworkWithReplace(Qubrick):
    """Assuming a pink localized enumeration, implement an improved fSWAP network to localize the plaquettes in parallel.

    As described in https://arxiv.org/abs/2012.09238.

    Note:
        - This is specifically designed for a pink happy enumeration, i.e., enumeration where the pink plaquettes are already localized
        - fswaps only need to be performed for the gold plaquettes.
        - It will execute a correct circuit for other enumerations but will throw a warning.
        - If `FermionicSwapWithReplace` is used, it will swap qubits `a` and `b`(with `a < b`) while restoring all intermediate qubits to their original order.
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
        """Implements the fSWAP network to localize the plaquettes.

        This qubrick takes in an enumeration and generates the appropriate colored plaquette indices in a default list order.

        Optimized to work with "pink happy" enumeration, i.e., enumeration where the pink plaquettes are already localized.
        Though, it will not error in the general case.

        Args:
            target_reg (Qubits): Register holding system qubits to apply the fSWAPs to.
            current_enumeration (List[np.darray]): Current enumeration of indices on the lattice
                                Note: it is a list to account for possible multiple spin sectors, e.g., pink-localized has `(up_enum, dn_enum)` where each has `L×L` plaquette indices.
            color_to_localize: Must be either 'pink' or 'gold', corresponding to the plaquette arrangement as described in https://arxiv.org/abs/2012.09238.

        Note:
            - Assuming the pink plaquettes are localized initially and the `color_to_localize` is pink, all swaps should be trivial and so nothing should happen.
            - But this is currently not hardcoded so that this qubrick can be called more generally
        """
        up_enumeration = current_enumeration[0]
        down_enumeration = current_enumeration[1]
        lattice_size = up_enumeration.shape[0]
        if color_to_localize == "pink":
            current_plaqs = get_plaqs([up_enumeration, down_enumeration], snaked=True)
        else:
            current_plaqs = get_plaqs(
                [shift_enumeration(up_enumeration), shift_enumeration(down_enumeration)], snaked=True
            )

        # Target enumeration is the same as pink happy
        pink_happy_up, pink_happy_down = generate_high_low_enum(lattice_size, pink_happy=True)
        target_plaqs = get_plaqs([pink_happy_up, pink_happy_down], snaked=True)

        fswap_list = get_fswap_with_replace_list(current_plaqs, target_plaqs, trivial_swaps=False, printing=False)

        if color_to_localize == "pink" and len(fswap_list) > 0:
            # fswap list is not empty and the plaquette is pink - this means the enumeration was not initially localized - raise a warning
            warnings.warn(
                "Pink localized fswap network is being used without the pink localized initial enumeration.",
                UserWarning,
            )

        for fswap in fswap_list:
            current_index = fswap[0]
            expected_index = fswap[1]
            # Perform the nonlocal swap
            self.fermionic_swap_qubrick.compute(target_reg, [current_index], [expected_index])

        # Flatten the plaquettes to feed back to the plaquette evolution
        localized_plaquettes = [
            [target_plaq[0][0], target_plaq[1][0], target_plaq[1][1], target_plaq[0][1]] for target_plaq in target_plaqs
        ]
        self.set_classical_result(localized_plaquettes, "localized_plaquettes")
