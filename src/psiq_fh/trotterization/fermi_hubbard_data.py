"""Dataclasses for Fermi-Hubbard systems."""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from ..utils.fermi_hubbard_hamiltonian import get_fermi_hubbard_hamiltonian
from ..utils.jw_ordering_utils import (
    generate_high_low_enum,
    generate_interaction_enumeration_indices,
    generate_plaquette_enumeration_indices,
    validate_enumerations,
)


@dataclass
class FermiHubbardData:
    """Dataclass for 2D Fermi-Hubbard model with nearest-neighbor hopping terms.

    This dataclass encapsulates all parameters needed to define a 2D Fermi-Hubbard
    system for quantum simulation. It automatically validates inputs and computes
    the Hamiltonian norm.

    Args:
        x_dim (int): Lattice width (must be even).
        y_dim (int): Lattice height (must be even).
        total_evolution_time (float): Total time for quantum evolution.
            Note: this is a pre-normalized time, the qubricks will divide by the one-norm of the Hamiltonian,
            so implements e^{-iH*total_evolution_time/(one_norm_of_H)}.
        n_trotter_steps (int): Number of Trotter steps for time evolution.
        enumeration (list or np.ndarray, optional): Site enumeration for spin-up
            and spin-down sectors. If None, uses high-low enumeration with simple snaking i.e. pink plaquettes not initially localised spin up sector has [0,..,L^2-2] fermionic mode labels, spin down sector has [L^2,...,2*L^2].
        u (float): Potential (interaction) coefficient. Defaults to 8.
        t (float): Kinetic (hopping) coefficient. Defaults to 1.
        particle_hole_symmetry (bool): Whether to use particle-hole symmetric
            Hamiltonian. Defaults to True.
        periodic (bool): Whether to use periodic boundary conditions.
            Defaults to True.
        spinless (bool): Whether to use spinless fermions. Defaults to False.

    Raises:
        ValueError: If u < 0, n_trotter_steps < 0, or lattice dimensions are odd.

    Note:
        - We assume kinetic_coefficient currently refers to its magnitude. The
          negative sign is assumed in methods.
        - If dimensions are 2x2, periodic boundary conditions override to False to avoid double counting edges in the plaquette construction.
    """

    x_dim: int
    y_dim: int
    total_evolution_time: float
    n_trotter_steps: int

    enumeration: list | np.ndarray = None
    u: float = 8  # potential coefficient
    t: float = 1  # hopping coefficient, negative sign of hopping assumed. i.e. t=1 corresponds to a hopping term with -a a^\dag.
    particle_hole_symmetry: bool = True
    periodic: bool = True
    spinless: bool = False

    def __post_init__(self):
        """Validate inputs and initialize default enumeration if needed.

        Raises:
            ValueError: If number of Trotter steps is negative,
            or lattice dimensions are odd.
        """
        if self.n_trotter_steps < 0:
            raise ValueError("Number of Trotter steps must be positive.")

        if (self.x_dim % 2 == 1) or (self.y_dim % 2 == 1):
            raise ValueError("Lattice size must be even.")

        if self.enumeration is None:
            self.enumeration = list(generate_high_low_enum(self.x_dim, self.y_dim))

        validate_enumerations(self.enumeration)

    @cached_property
    def norm(self) -> float:
        """Compute the one-norm of the Fermi-Hubbard Hamiltonian.

        Returns:
            float: The one-norm of the Hamiltonian, cached after first computation.
        """
        if self.x_dim == 2 and self.y_dim == 2:
            # override to use non-periodic boundaries for the 2x2 case, to avoid double
            # counting edges in the plaquette construction
            self.periodic = False

        return get_fermi_hubbard_hamiltonian(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            potential_coefficient=self.u,
            kinetic_coefficient=self.t,
            periodic=self.periodic,
            particle_hole_symmetry=self.particle_hole_symmetry,
            spinless=self.spinless,
        ).norm()


@dataclass
class InteractionTermData:
    """Dataclass for interaction (potential) terms in the Fermi-Hubbard model.

    This dataclass handles the U-term (on-site repulsion) part of the Fermi-Hubbard
    Hamiltonian, providing the necessary data for Trotterized time evolution.

    Args:
        enumeration (list or np.ndarray): Site enumeration arrays for spin-up
            and spin-down sectors.
        evolution_time (float): Time duration for evolving the interaction term.
        coefficient (float): Hamiltonian coefficient
            used for rotation angle calculation.
        particle_hole_symmetry (bool, optional): Whether to use particle-hole
            symmetric Hamiltonian. Defaults to True if not specified.

    Attributes:
        interaction_indices (np.ndarray): Computed indices for ZZ interactions.
    """

    enumeration: list | np.ndarray
    evolution_time: float
    coefficient: float
    particle_hole_symmetry: bool | None = None

    def __post_init__(self):
        """Validate enumeration and set default particle-hole symmetry.

        Raises:
            ValueError: If enumeration is invalid.
        """
        validate_enumerations(self.enumeration)

        if self.particle_hole_symmetry is None:
            self.particle_hole_symmetry = True

    @property
    def interaction_indices(self) -> np.ndarray:
        """Qubit indices for ZZ interaction gates.

        Returns:
            np.ndarray: Array of tuples (i, j) representing qubit pairs for
                ZZ interactions in the interaction term.
        """
        return generate_interaction_enumeration_indices(self.enumeration)


@dataclass
class PlaquetteTermData:
    """Dataclass for plaquette (hopping) terms in the Fermi-Hubbard model.

    This dataclass handles the kinetic energy terms in the Fermi-Hubbard Hamiltonian
    using the plaquette-based Trotterization scheme from arXiv:2012.09238.

    Args:
        color (Literal["pink", "gold"]): Plaquette color in the checkerboard pattern.
            Determines which plaquettes are evolved simultaneously.
        enumeration (list or np.ndarray): Site enumeration arrays for spin-up
            and spin-down sectors.
        evolution_time (float): Time duration for evolving the plaquette term.
        coefficient (float): Hamiltonian coefficient
            used for rotation angle calculation.

    Attributes:
        plaquette_indices (np.ndarray): Computed indices for plaquette operations.

    Raises:
        ValueError: If color is not "pink" or "gold", or enumeration is invalid.
    """

    color: Literal["pink", "gold"]
    enumeration: list | np.ndarray
    evolution_time: float
    coefficient: float

    def __post_init__(self):
        """Validate enumeration and plaquette color.

        Raises:
            ValueError: If enumeration is invalid or color is not "pink" or "gold".
        """
        validate_enumerations(self.enumeration)

        if self.color not in ["pink", "gold"]:
            raise ValueError(f"Invalid color '{self.color}'; must be 'pink' or 'gold'.")

    @property
    def plaquette_indices(self) -> np.ndarray:
        """Site indices for plaquette operations.

        Returns:
            np.ndarray: Array of plaquette site indices for the specified color,
                arranged for efficient quantum circuit implementation.
        """
        return generate_plaquette_enumeration_indices(self.enumeration, self.color)
