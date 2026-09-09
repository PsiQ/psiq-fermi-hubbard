"""Allocate catalyst_registers for vanilla Fermi-Hubbard plaquette Trotterization."""

import numpy as np


def allocate_catalyst_registers_IPG(
    x_dim, y_dim, potential_coefficient, kinetic_coefficient, total_evolution_time, n_trotter_steps, n_hwp_batches, qc
):
    """Allocate all catalysts from plaquette Trotterized time evolution operator that uses generalized
       phase gradient addition for implementing Hamming weight phasing. Assumes IPG ordering.

    Args:
        x_dim (int): x-dimension of Fermi-Hubbard lattice, even only
        y_dim (int): y-dimension of Fermi-Hubbard lattice, even only
        potential_coefficient (float): onsite interaction strength
        kinetic_coefficient (float): hopping interaction strength
        total_evolution_time (float): total evolution time of operator
        n_trotter_steps (int): number of Trotter steps
        n_hwp_batches (int): number of batches in Hamming weight phasing
        qc (QPU): qpu

    Notes:
       - Should be 2 floor(log2(x_dim * y_dim / n_hwp_batches)) + 4 rotations in total.
    """
    # Evolution time per Trotter step
    single_trotter_step_evolution_time = total_evolution_time / n_trotter_steps

    # For all terms: multiplied by 2 because the tower of rotations is Rz
    #   (which has a factor of 1/2 in gate definition)
    #   but catalyst rotations are of Phase type

    # For unmerged interaction term
    # Divide potential coefficient u by 4 from JW
    # Divide evolution time by 2 for second order
    angle = -1 * potential_coefficient * single_trotter_step_evolution_time * 2 * 180 / np.pi / 4 / 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg1:
        pass

    # For merged interaction term
    # Divide potential coefficient u by 4 from JW
    # Divide evolution time by 2 for second order
    # Multiplied by 2 for merging
    angle = -1 * potential_coefficient * single_trotter_step_evolution_time * 2 * 180 / np.pi / 4 / 2 * 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg2:
        pass

    # For pink plaquette term
    # Divide evolution time by 2 for second order
    angle = 1 * 2 * single_trotter_step_evolution_time * kinetic_coefficient * 180 / np.pi / 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg3:
        pass

    # For merged gold term
    # Multiplied by 2 for second order
    angle = 1 * 2 * single_trotter_step_evolution_time * kinetic_coefficient * 180 / np.pi / 2 * 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg3:
        pass


def allocate_catalyst_registers_PIG(
    x_dim, y_dim, potential_coefficient, kinetic_coefficient, total_evolution_time, n_trotter_steps, n_hwp_batches, qc
):
    """Allocate all catalysts from plaquette Trotterized time evolution operator that uses generalized
       phase gradient addition for implementing Hamming weight phasing. Assumes PIG ordering.

    Args:
        x_dim (int): x-dimension of Fermi-Hubbard lattice, even only
        y_dim (int): y-dimension of Fermi-Hubbard lattice, even only
        potential_coefficient (float): onsite interaction strength
        kinetic_coefficient (float): hopping interaction strength
        total_evolution_time (float): total evolution time of operator
        n_trotter_steps (int): number of Trotter steps
        n_hwp_batches (int): number of batches in Hamming weight phasing
        qc (QPU): qpu

    Notes:
       - Should be 2 floor(log2(x_dim * y_dim / n_hwp_batches)) + 3 rotations in total.
    """
    # Evolution time per Trotter step
    single_trotter_step_evolution_time = total_evolution_time / n_trotter_steps

    # For all terms: multiplied by 2 because the tower of rotations is Rz
    #   (which has a factor of 1/2 in gate definition)
    #   but catalyst rotations are of Phase type

    # For unmerged pink plaquette term
    # Divided by 2 because of evolution time is divided by 2 in second order
    # Converted to degrees via 180/pi
    angle = 1 * 2 * single_trotter_step_evolution_time * kinetic_coefficient * 180 / np.pi / 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg1:
        pass

    # For interaction term
    # Divide potential coefficient u by 4 from JW
    # Multiply the evolution time by 2 for second order
    angle = -1 * potential_coefficient * single_trotter_step_evolution_time * 2 * 180 / np.pi / 4 / 2
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg2:
        pass

    # For gold and merged pink plaquette terms
    # Implicit but from the unmerged, multiplied by 2
    angle = 1 * 2 * single_trotter_step_evolution_time * kinetic_coefficient * 180 / np.pi
    size_of_catalyst_state = int(np.floor(np.log2(x_dim * y_dim / n_hwp_batches))) + 1
    cat_angle = (1 << (size_of_catalyst_state - 1)) * angle
    with qc.fetch_rotation_catalyst_state(cat_angle, size_of_catalyst_state) as catalyst_state_reg3:
        pass


def fetch_catalyst_qubits(qc):
    """Return catalyst qubits as a register.

    Args:
        qc (QPU): QPU where catalysts are called

    Returns:
        catalyst_reg (Qubits): Register of catalyst qubits
    """
    catalyst_qubit_addresses = list(qc._rotation_catalyst_qubits.values())
    catalyst_reg = catalyst_qubit_addresses[0][0]
    for qbit, _ in catalyst_qubit_addresses[1:]:
        catalyst_reg |= qbit
    return catalyst_reg
