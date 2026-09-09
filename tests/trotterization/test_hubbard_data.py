"""Tests for dataclasses for Fermi-Hubbard systems."""

import numpy as np
import pytest

from psiq_fh.trotterization.fermi_hubbard_data import (
    InteractionTermData,
    PlaquetteTermData,
    FermiHubbardData,
)
from psiq_fh.utils.fermi_hubbard_hamiltonian import get_expected_norm


def setup_valid_vanilla_fh_data():
    fh_data = FermiHubbardData(
        x_dim=4,
        y_dim=4,
        total_evolution_time=np.pi,
        n_trotter_steps=10,
        enumeration=None,  # default
        particle_hole_symmetry=True,
        t=1,
        u=4,
    )
    return fh_data


def test_invalid_inputs_fh_data():
    # Odd dim
    with pytest.raises(ValueError, match=r"Lattice size must be even."):
        FermiHubbardData(
            x_dim=1,
            y_dim=4,
            total_evolution_time=np.pi,
            n_trotter_steps=10,
        )

    # Negative number of Trotter steps
    with pytest.raises(ValueError, match=r"Number of Trotter steps must be positive."):
        FermiHubbardData(
            x_dim=4,
            y_dim=4,
            total_evolution_time=np.pi,
            n_trotter_steps=-10,
        )

    # Valid case: zero potential coefficient (useful for turning off
    # and on parts of Hamiltonian)
    valid_fh_data = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=np.pi, n_trotter_steps=10, u=0)

    # Duplicated index within a single spin sector
    duplicated_within_sector = [np.array([[0, 1], [1, 2]])]  # '1' is duplicated within the array
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        FermiHubbardData(
            x_dim=4,
            y_dim=4,
            total_evolution_time=np.pi,
            n_trotter_steps=10,
            enumeration=duplicated_within_sector,
        )

    # Duplicate across arrays
    duplicated_across_sector = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 5], [3, 6]]),  # '3' appears in both arrays
    ]
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        FermiHubbardData(
            x_dim=4,
            y_dim=4,
            total_evolution_time=np.pi,
            n_trotter_steps=10,
            enumeration=duplicated_across_sector,
        )


def test_invalid_inputs_plaquette_data():
    # color not pink nor gold
    valid_color = "pink"
    valid_enumeration = [np.array([[0, 3], [1, 2]])]
    valid_time = 1
    valid_coeff = 1

    with pytest.raises(ValueError, match=r"Invalid color 'orange'; must be 'pink' or 'gold'."):
        PlaquetteTermData("orange", valid_enumeration, valid_time, valid_coeff)

    # Duplicated index within a single spin sector
    duplicated_within_sector = [np.array([[0, 1], [1, 2]])]
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        duplicated_within_sector = [np.array([[0, 1], [1, 2]])]
        PlaquetteTermData(valid_color, duplicated_within_sector, valid_time, valid_coeff)

    # Duplicate across arrays
    duplicated_across_sector = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 5], [3, 6]]),  # '3' appears in both arrays
    ]
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        PlaquetteTermData(valid_color, duplicated_across_sector, valid_time, valid_coeff)


def test_invalid_inputs_interaction_data():
    valid_time = 1
    valid_norm_coeff = 1

    # Duplicated index within a single spin sector
    duplicated_within_sector = [np.array([[0, 1], [1, 2]])]
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        InteractionTermData(duplicated_within_sector, valid_time, valid_norm_coeff)

    # Duplicate across arrays
    duplicated_across_sector = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 5], [3, 6]]),  # '3' appears in both arrays
    ]
    with pytest.raises(ValueError, match=r"Duplicated indices found in enumerations."):
        InteractionTermData(duplicated_across_sector, valid_time, valid_norm_coeff)


def test_vanilla_defaults_and_properties():
    """Test default values and computed properties of Vanilla2DFermiHubbardData."""
    # Test default values
    fh_data = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=1.0, n_trotter_steps=5)

    # Check defaults
    assert fh_data.u == 8
    assert fh_data.t == 1
    assert fh_data.particle_hole_symmetry == True
    assert fh_data.periodic == True
    assert fh_data.spinless == False
    assert fh_data.enumeration is not None  # Should auto-generate

    # norm is a cached_property, so nothing is computed until it is first read
    assert "norm" not in fh_data.__dict__

    # Test norm property computation
    norm_value = fh_data.norm
    assert isinstance(norm_value, (int, float))
    assert norm_value > 0

    # ...and is cached on the instance afterwards
    assert "norm" in fh_data.__dict__

    # Test caching behavior - should return same value
    assert fh_data.norm == norm_value

    # Test we get the expected norm value
    expected_norm = get_expected_norm(
        L_x=4,
        L_y=4,
        potential_coefficient=fh_data.u,
        kinetic_coefficient=fh_data.t,
        particle_hole_symmetry=fh_data.particle_hole_symmetry,
    )
    assert np.isclose(norm_value, expected_norm)


def test_interaction_term_data_properties():
    """Test InteractionTermData computed properties and defaults."""
    enumeration = [np.array([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7]])]

    # Test with explicit particle_hole_symmetry
    interaction_data = InteractionTermData(
        enumeration=enumeration, evolution_time=0.5, coefficient=2.0, particle_hole_symmetry=False
    )
    assert interaction_data.particle_hole_symmetry == False

    # Test interaction_indices property
    indices = interaction_data.interaction_indices
    assert isinstance(indices, (np.ndarray, list))
    assert len(indices) > 0  # Should have some interactions

    # Test default particle_hole_symmetry
    interaction_data_default = InteractionTermData(enumeration=enumeration, evolution_time=0.5, coefficient=2.0)
    assert interaction_data_default.particle_hole_symmetry == True


def test_plaquette_term_data_properties():
    """Test PlaquetteTermData computed properties and color validation."""
    enumeration = [np.array([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7]])]

    # Test pink plaquettes
    pink_data = PlaquetteTermData(color="pink", enumeration=enumeration, evolution_time=0.25, coefficient=-2.0)
    assert pink_data.color == "pink"

    # Test plaquette_indices property
    pink_indices = pink_data.plaquette_indices
    assert isinstance(pink_indices, (np.ndarray, list))

    # Test gold plaquettes
    gold_data = PlaquetteTermData(color="gold", enumeration=enumeration, evolution_time=0.25, coefficient=-2.0)

    gold_indices = gold_data.plaquette_indices
    assert isinstance(gold_indices, (np.ndarray, list))


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test minimum lattice size
    min_data = FermiHubbardData(x_dim=2, y_dim=2, total_evolution_time=0.1, n_trotter_steps=1)
    assert min_data.x_dim == 2
    assert min_data.y_dim == 2

    # Test zero evolution time (should be valid)
    zero_time_data = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=0.0, n_trotter_steps=1)
    assert zero_time_data.total_evolution_time == 0.0

    # Test single Trotter step
    single_step = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=1.0, n_trotter_steps=1)
    assert single_step.n_trotter_steps == 1


def test_different_parameter_combinations():
    """Test various parameter combinations for comprehensive coverage."""
    # Test spinless system
    spinless_data = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=1.0, n_trotter_steps=5, spinless=True)
    assert spinless_data.spinless == True

    # Test non-periodic boundary conditions
    open_bc_data = FermiHubbardData(x_dim=4, y_dim=4, total_evolution_time=1.0, n_trotter_steps=5, periodic=False)
    assert open_bc_data.periodic == False

    # Test particle-hole asymmetric
    ph_asym_data = FermiHubbardData(
        x_dim=4, y_dim=4, total_evolution_time=1.0, n_trotter_steps=5, particle_hole_symmetry=False
    )
    assert ph_asym_data.particle_hole_symmetry == False
