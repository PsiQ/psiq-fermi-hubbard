"""Tests for Fermi-Hubbard Hamiltonian utility functions."""

import numpy as np
from openfermion import get_sparse_operator
from psiqdk.algorithms.utils import pauli_sum_to_numpy
from psiqdk.workbench.utils.numpy_utils import fidelity

from psiq_fh.utils.fermi_hubbard_hamiltonian import (
    construct_kinetic_hamiltonian_from_edges,
    construct_potential_hamiltonian,
    get_fermi_hubbard_hamiltonian,
)


def test_psiq_and_openfermion_fh_hamiltonian_constructions_for_2x2():
    """Test that PsiQ's FH Hamiltonian representation and OpenFermion's
    produce the same energies and states.

    Note:
        - for PsiQ, we set kinetic coefficient as -1
        - for method we wrote on top of OpenFermion, this negative sign
          is hard-coded, thus we input +1
    """
    lattice_size = 2
    potential_coefficient = 8
    particle_hole_symmetry = True

    # PsiQ
    hubbard_hamiltonian = get_fermi_hubbard_hamiltonian(
        lattice_size,
        lattice_size,
        potential_coefficient,
        kinetic_coefficient=-1,
        particle_hole_symmetry=particle_hole_symmetry,
    )
    hubbard_hamiltonian_matrix = pauli_sum_to_numpy(hubbard_hamiltonian)
    hubbard_energies_psiq, hubbard_vectors_psiq = np.linalg.eigh(hubbard_hamiltonian_matrix)

    # OpenFermion
    edges = [(0, 2), (2, 6), (6, 4), (4, 0), (1, 5), (5, 7), (7, 3), (3, 1)]
    # Start with the kinetic Hamiltonian
    constructed_hamiltonian = construct_kinetic_hamiltonian_from_edges(edges, kinetic_coeff=1)

    # Add onsite interaction
    constructed_hamiltonian += construct_potential_hamiltonian(
        x_dim=2, y_dim=2, potential_coeff=potential_coefficient, particle_hole_symmetry=particle_hole_symmetry
    )

    constructed_hamiltonian = get_sparse_operator(constructed_hamiltonian).todense()

    hubbard_energies_of, hubbard_vectors_of = np.linalg.eigh(constructed_hamiltonian)

    # Check energies
    assert np.allclose(hubbard_energies_psiq, hubbard_energies_of)

    # They differ in ordering (big vs. little endian)
    flipped = np.flipud(np.fliplr(hubbard_hamiltonian_matrix)).T
    assert np.allclose(flipped, constructed_hamiltonian)

    # Check ground state
    gstate_psiq = np.asarray(np.flip(hubbard_vectors_psiq[:, 0]))  # reversed order
    gstate_of = np.asarray(hubbard_vectors_of[:, 0]).flatten()
    assert np.isclose(fidelity(gstate_psiq, gstate_of), 1)


def test_alt_enumerations_return_same_energy():
    """Test that the alternative enumerations return the same system energy."""
    potential_coefficient = 8
    particle_hole_symmetry = True

    # Even-odd
    edges = [(0, 2), (2, 6), (6, 4), (4, 0), (1, 5), (5, 7), (7, 3), (3, 1)]
    kinetic_hamiltonian_even_odd = construct_kinetic_hamiltonian_from_edges(edges, kinetic_coeff=1)
    potential_hamiltonian_even_odd = construct_potential_hamiltonian(
        x_dim=2,
        y_dim=2,
        potential_coeff=potential_coefficient,
        particle_hole_symmetry=particle_hole_symmetry,
        even_odd=True,
    )
    constructed_hamiltonian = kinetic_hamiltonian_even_odd + potential_hamiltonian_even_odd
    constructed_hamiltonian_even_odd = get_sparse_operator(constructed_hamiltonian).todense()
    hubbard_energies_even_odd, _ = np.linalg.eigh(constructed_hamiltonian_even_odd)

    # High-low
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    kinetic_hamiltonian_high_low = construct_kinetic_hamiltonian_from_edges(edges, kinetic_coeff=1)
    potential_hamiltonian_high_low = construct_potential_hamiltonian(
        x_dim=2,
        y_dim=2,
        potential_coeff=potential_coefficient,
        particle_hole_symmetry=particle_hole_symmetry,
        even_odd=False,
    )
    constructed_hamiltonian = kinetic_hamiltonian_high_low + potential_hamiltonian_high_low
    constructed_hamiltonian_high_low = get_sparse_operator(constructed_hamiltonian).todense()
    hubbard_energies_high_low, _ = np.linalg.eigh(constructed_hamiltonian_high_low)

    assert np.allclose(hubbard_energies_even_odd, hubbard_energies_high_low)
