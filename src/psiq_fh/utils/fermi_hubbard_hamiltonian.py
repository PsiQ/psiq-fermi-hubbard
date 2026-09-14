"""Utility functions for generating and analyzing Fermi-Hubbard Hamiltonians."""

import numpy as np
from openfermion import get_sparse_operator
from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from psiqdk.algorithms.utils.paulimask import PauliMask, PauliSum


def get_expected_norm(
    L_x: int,
    L_y: int,
    potential_coefficient: float,
    kinetic_coefficient: float = 1,
    particle_hole_symmetry: bool = False,
) -> float:
    """Analytically calculate the norm of the Fermi-Hubbard Hamiltonian as defined by Eq. 57 of arXiv:1805.03662.

    Args:
        L_x: Physical lattice x dimension of the system in question.
        L_y: Physical lattice y dimension of the system in question.
        potential_coefficient: The strength of the potential (coulomb) interaction (typically denoted by ``u``)
        kinetic_coefficient: The strength of the kinetic (hopping) interaction (typically denoted by
            ``t``). Defaults to 1.
        particle_hole_symmetry: Whether the Hamiltonian has particle hole symmetry or not. Defaults to False.

    Note:
        This assumed periodic boundary conditions to keep in line with
        the form used in arXiv:1805.03662.

    Returns:
        The expected norm as a float (positive and real-valued).
    """
    n_sites = L_x * L_y

    # Potential term
    if particle_hole_symmetry:
        # In Jordan-Wigner, the operator norm of n_up n_down is 1/4
        # only one spin sector contributes at half-filling
        potential_term = potential_coefficient * n_sites / 4
    else:
        # both spin sectors contribute
        potential_term = 3 * potential_coefficient * n_sites / 4

    # Kinetic terms
    # Count horizontal (x) links per spin sector:
    # - If L_x > 2, each row has L_x links (due to PBC)
    # - If L_x == 2, only one unique horizontal link per row
    x_links_per_spin = L_y * (1 if L_x == 2 else L_x)

    # Count vertical (y) links per spin sector:
    # - If L_y > 2, each column has L_y links (due to PBC)
    # - If L_y == 2, only one unique vertical link per column
    y_links_per_spin = L_x * (1 if L_y == 2 else L_y)

    # Total links across both spin sectors (2 spins)
    total_links = 2 * (x_links_per_spin + y_links_per_spin)
    # Each hopping term has norm 1 (from c^dagger_i c_j + h.c.)
    kinetic_term = kinetic_coefficient * total_links

    return potential_term + kinetic_term


def get_fermi_hubbard_hamiltonian(
    x_dim: int,
    y_dim: int,
    potential_coefficient: float,
    kinetic_coefficient: float = 1,
    periodic: bool = True,
    particle_hole_symmetry: bool = False,
    spinless: bool = False,
):
    """Wrapper function around openfermion to return the Fermi Hubbard Hamiltonian as a ``PauliSum``.

    Args:
        x_dim: The x-dimension of the physical lattice in the target system.
        y_dim: The y-dimension of the physical lattice in the target system.
        potential_coefficient: The strength of the potential (coulomb) interaction (typically denoted by ``u``).
        kinetic_coefficient: The strength of the kinetic (hopping) interaction (typically denoted by
            ``t``). Defaults to 1.
        periodic: If ``True``, looks at the system using periodic boundary conditions. Defaults to
            ``True``.
        particle_hole_symmetry: If ``True``, uses the particle-hole symmetric hamiltonian. Defaults to
            ``False``.
        spinless: If ``True``, considers a spinless Hamiltonian. Defaults to ``False``.

    Returns:
        ``PauliSum`` representing the Fermi-Hubbard Hamiltonian as defined in Eq. 57 of arXiv:1805.03662 with the added
        caveat of removing the identity term from the Hamiltonian.

    """
    # Note, this Hamiltonian is not projected into a particular particle number subspace
    unprojected_openfermion_hamiltonian = fermi_hubbard(
        x_dim,
        y_dim,
        tunneling=kinetic_coefficient,
        coulomb=potential_coefficient,
        periodic=periodic,
        particle_hole_symmetry=particle_hole_symmetry,
        spinless=spinless,
    )
    qubit_hamiltonian = jordan_wigner(unprojected_openfermion_hamiltonian)

    hubbard_coefficients, hubbard_operators = [], []
    terms = qubit_hamiltonian.terms
    for term in terms:
        hubbard_coefficients.append(terms[term].real)
        hubbard_operators.append(term)

    hubbard_hamiltonian = PauliSum()
    for coefficient, multi_op in zip(hubbard_coefficients, hubbard_operators):
        op_string = ""
        for single_op in multi_op:
            op_string += single_op[1] + str(single_op[0]) + " "

        pauli_mask = PauliMask.from_pauli_string(op_string)

        if not (pauli_mask.mask[0] == 0 and pauli_mask.mask[1] == 0):  # Ignore Identity Term
            hubbard_hamiltonian += PauliSum([coefficient, pauli_mask])

    return hubbard_hamiltonian


def construct_kinetic_hamiltonian_from_edges(edges: list[tuple[int, int]], kinetic_coeff: float = 1) -> QubitOperator:
    """Constructs the kinetic (hopping) fermionic Hamiltonian for a given list of edges
    in a graph, and returns its Jordan-Wigner transformed qubit representation.

    Each edge (i, j) contributes two terms to the Hamiltonian:
        - - kinetic_coeff * a_i† a_j  (fermion hops from j to i)
        - - kinetic_coeff * a_j† a_i  (fermion hops from i to j)

    Args:
        edges (List[Tuple[int, int]]): A list of (i, j) tuples representing undirected edges
                                       in the hopping graph.
        kinetic_coeff (float, optional): Coefficient for the kinetic (hopping) term.
                                         Defaults to 1.0.

    Returns:
        FermionOperator: The Jordan-Wigner transformed qubit Hamiltonian representing
                         the kinetic hopping terms across the given edges.
    """
    fermionic_ham = FermionOperator()

    for i, j in edges:
        fermionic_ham += FermionOperator(f"{i}^ {j}", -kinetic_coeff)
        fermionic_ham += FermionOperator(f"{j}^ {i}", -kinetic_coeff)

    return jordan_wigner(fermionic_ham)


def construct_potential_hamiltonian(
    potential_coeff: float, x_dim: int, y_dim: int, particle_hole_symmetry: bool = False, even_odd: bool = True
) -> QubitOperator:
    """Construct onsite interaction term and return in qubit representation, for the 2x2 square Fermi-Hubbard model.

    Args:
        potential_coeff: Coefficient for the interaction term.
        x_dim: The x-dimension of the physical lattice in the target system.
        y_dim: The y-dimension of the physical lattice in the target system.
        particle_hole_symmetry: If ``True``, uses the particle-hole
            symmetric Hamiltonian. Defaults to ``False``.
        even_odd: Defaults to True, and spin up being even numbers, spin down being odd.
            If False adopts the high-low configuration with spin up low number, spin down high.
    """
    potential_hamiltonian = QubitOperator()
    num_sites = x_dim * y_dim
    for site in range(num_sites):
        if even_odd:
            potential_hamiltonian += QubitOperator(f"Z{2 * site} Z{2 * site + 1}", potential_coeff / 4)
            if not particle_hole_symmetry:
                potential_hamiltonian += QubitOperator(f"Z{2 * site}", -potential_coeff / 4)
                potential_hamiltonian += QubitOperator(f"Z{2 * site + 1}", -potential_coeff / 4)
        else:
            # high-low spin ordering
            potential_hamiltonian += QubitOperator(f"Z{site} Z{site + num_sites}", potential_coeff / 4)
            if not particle_hole_symmetry:
                potential_hamiltonian += QubitOperator(f"Z{site}", -potential_coeff / 4)
                potential_hamiltonian += QubitOperator(f"Z{site + num_sites}", -potential_coeff / 4)
    return potential_hamiltonian


def edges_from_enumeration(enumeration: np.ndarray) -> list[tuple[int, int]]:
    """Generate a list of edges assuming periodic boundary conditions from an enumeration.

    Args:
        enumeration (np.ndarray): A 2D numpy array where each entry represents the
                                  label of a site in the lattice.

    Returns:
        List[Tuple[int, int]]: A list of undirected edges represented as tuples of
                               site indices. Each edge appears only once (no duplicates).

    Example:
        enum = np.array([[0, 3, 4, 7],
                         [1, 2, 5, 6]])
        edges = [(0, 3), (3, 4), (4, 7), (7, 0), (1, 2), (2, 5), (5, 6), (6, 1), (0, 1), (3, 2), (4, 5), (7, 6), (2, 3), (5, 4), (6, 7), (1, 0)].
    """
    rows, cols = enumeration.shape
    edges = []

    for i in range(rows):
        for j in range(cols):
            current = enumeration[i, j]
            right = enumeration[i, (j + 1) % cols]
            down = enumeration[(i + 1) % rows, j]
            edges.append((current, right))
            edges.append((current, down))

    return edges


def get_fermi_hubbard_hamiltonian_2x2_even_odd(
    potential_coefficient: float = 8, kinetic_coefficient: float = 1, particle_hole_symmetry: bool = True
) -> np.ndarray:
    """Create a numpy array Fermi Hubbard Hamiltonian for the 2x2 case with even odd enumeration.

    Args:
        potential_coefficient: The strength of the potential (coulomb) interaction (typically denoted by ``u``)
        kinetic_coefficient: The strength of the kinetic (hopping) interaction (typically denoted by
            ``t``). Defaults to 1.
        particle_hole_symmetry: Whether the Hamiltonian has particle hole symmetry or not. Defaults to False.

    Returns:
        np.array of Fermi Hubbard Hamiltonian.

    Note:
    Spin up:        Spin down:
        0-2             1-3
        | |             | |
        4-6             5-7
    """
    edges = [(0, 2), (2, 6), (6, 4), (4, 0), (1, 5), (5, 7), (7, 3), (3, 1)]
    # Start with the kinetic Hamiltonian
    constructed_hamiltonian = construct_kinetic_hamiltonian_from_edges(edges, kinetic_coeff=kinetic_coefficient)

    # Add onsite interaction
    constructed_hamiltonian += construct_potential_hamiltonian(
        x_dim=2, y_dim=2, potential_coeff=potential_coefficient, particle_hole_symmetry=particle_hole_symmetry
    )

    constructed_hamiltonian = get_sparse_operator(constructed_hamiltonian).todense()
    return constructed_hamiltonian
