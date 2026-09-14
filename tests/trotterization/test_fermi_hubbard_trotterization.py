"""Tests for the Fermi-Hubbard Trotterization circuit in arxiv:2012.09238."""

import numpy as np
import pytest
from openfermion import get_sparse_operator
from psiqdk.algorithms import ComputeHammingWeightGroupOfThrees, PowerOfTwoBatchedHammingWeightPhasing
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.utils.numpy_utils import fidelity, reverse_numpy_op
from scipy.linalg import expm


from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.trotterization.fermi_hubbard_trotterization import (
    HubbardPlaquetteTrotterizationIPG,
    HubbardPlaquetteTrotterizationPIG,
)
from psiq_fh.trotterization.fswapping.fermionic_swap import FermionicSwapWithReplaceAVOpt
from psiq_fh.utils.jw_ordering_utils import generate_high_low_enum
from psiq_fh.trotterization.hopping import PlaquetteTrotterStep, TwoModeFFFTViaPPRs, exptXXYY
from psiq_fh.trotterization.interaction import InteractionTrotterStep
from psiq_fh.utils.fermi_hubbard_hamiltonian import (
    construct_kinetic_hamiltonian_from_edges,
    edges_from_enumeration,
    get_fermi_hubbard_hamiltonian_2x2_even_odd,
)
from psiq_fh.utils.control_qubit import DirectionalControlQubit
from psiq_fh.utils.jw_ordering_utils import (
    generate_even_odd_enum,
)


from psiq_fh.trotterization.hopping import exptXXYY
from psiq_fh.trotterization.directional_hamming_weight_phasing import (
    PowerOfTwoBatchedDirectionalHammingWeightPhasing,
)
from psiq_fh.trotterization.catalyst_allocation import (
    allocate_catalyst_registers_PIG,
    fetch_catalyst_qubits,
)


@pytest.mark.parametrize(
    "potential_coefficient, kinetic_coefficient, total_evolution_time, number_of_trotter_steps",
    [
        (0, 4.2, 0.6, 1),
        (7.3, 0, 1.3, 1),
        (8, 1, 0.4, 10),
        (
            6.2,
            1.5,
            0.4,
            10,
        ),  # Note: due to fixed error have to set no trotter steps sufficiently high to be a good approximation.
    ],
)
@pytest.mark.parametrize("particle_hole_symmetry", [True, False])
@pytest.mark.parametrize("trotter_qubrick", [HubbardPlaquetteTrotterizationIPG, HubbardPlaquetteTrotterizationPIG])
def test_action_of_HubbardPlaquetteTrotterization_for_2x2_lattice(
    potential_coefficient,
    kinetic_coefficient,
    total_evolution_time,
    number_of_trotter_steps,
    particle_hole_symmetry,
    trotter_qubrick,
):
    """Test the action of the HubbardPlaquetteTrotterization qubrick for a 2x2 lattice.

    Notes:
        - In the 2x2 case we do not include periodic boundary conditions, as an edge case:
        under PBC the gold plaquette coincides with the pink one, and this fixture has no
        gold evolution to pair with it.
        - Defaulting to even/odd enumeration.
    """
    x_dim = 2
    y_dim = 2
    num_state_qubits = 2 * (x_dim * y_dim)

    # For testing use even odd enumeration where up (down) spin sector has even (odd) fermionic mode numbering.
    constructed_hamiltonian = get_fermi_hubbard_hamiltonian_2x2_even_odd(
        potential_coefficient=potential_coefficient,
        kinetic_coefficient=kinetic_coefficient,
        particle_hole_symmetry=particle_hole_symmetry,
    )

    time_evolution = reverse_numpy_op(expm(total_evolution_time * 1j * constructed_hamiltonian))

    qc = QPU(num_qubits=num_state_qubits)

    psi_reg = Qubits(num_state_qubits, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=generate_even_odd_enum(x_dim, y_dim),  # Use even-odd splitting for testing
        particle_hole_symmetry=particle_hole_symmetry,
        t=kinetic_coefficient,
        u=potential_coefficient,
    )
    hubbard_time_evolution = trotter_qubrick(InteractionTrotterStep(), PlaquetteTrotterStep(), PlaquetteTrotterStep())

    hubbard_time_evolution.compute(psi_reg, fh_data)
    final_state = qc.pull_state()

    # Note: The two states differ by a small but non-zero global phase, so np.allclose on the
    # state vectors fails at some evolution times while fidelity does not. Bounding this
    # tolerance against the Trotter error is an open question upstream.
    # assert np.allclose(final, time_evolution @ initial) will currently fail
    # Some tolerance 1e-4 here taking the place of Trotter error analysis
    assert abs(fidelity(final_state, time_evolution @ initial) - 1) < 1e-4


@pytest.mark.parametrize(
    "total_evolution_time",
    [np.pi, 1],
)
@pytest.mark.parametrize("trotter_qubrick", [HubbardPlaquetteTrotterizationIPG, HubbardPlaquetteTrotterizationPIG])
def test_action_of_HubbardPlaquetteTrotterization_for_2x4_lattice_on_kinetic_hamiltonian(
    total_evolution_time, trotter_qubrick
):
    """Test the action of the HubbardPlaquetteTrotterization qubrick for a 2x4 lattice with kinetic only Hamiltonian."""
    x_dim = 2
    y_dim = 4
    num_state_qubits = x_dim * y_dim

    # First get the hamiltonian norm - we can reuse this helper function with potential coeff = 0
    potential_coefficient = 0

    # We only have one spin sector
    spin_up = np.array([[0, 3, 4, 7], [1, 2, 5, 6]])
    edges = edges_from_enumeration(spin_up)

    kin_coefficient = 1.34
    kin_ham = construct_kinetic_hamiltonian_from_edges(edges, kinetic_coeff=kin_coefficient)

    constructed_hamiltonian = get_sparse_operator(kin_ham).todense()
    time_evolution = reverse_numpy_op(expm(total_evolution_time * 1j * constructed_hamiltonian))

    qc = QPU(num_qubits=num_state_qubits)

    psi_reg = Qubits(num_state_qubits, "psi", qc)

    qc.set_param("random_seed", 42)
    qc.set_random()
    initial = qc.pull_state()

    number_of_trotter_steps = 1

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=[spin_up],
        particle_hole_symmetry=True,
        t=kin_coefficient,
        u=potential_coefficient,
    )

    hubbard_time_evolution = trotter_qubrick(InteractionTrotterStep(), PlaquetteTrotterStep(), PlaquetteTrotterStep())

    hubbard_time_evolution.compute(psi_reg, fh_data)
    final_state = qc.pull_state()

    # There is no error in the evolution
    assert np.allclose(fidelity(final_state, time_evolution @ initial), 1)
    assert np.allclose(final_state, time_evolution @ initial)


@pytest.mark.parametrize("potential_coefficient", [0.01, 11.23, 17.8])
@pytest.mark.parametrize("total_evolution_time, number_of_trotter_steps", [(1 << 0, 1)])
@pytest.mark.parametrize("trotter_qubrick", [HubbardPlaquetteTrotterizationIPG, HubbardPlaquetteTrotterizationPIG])
def test_hubbard_trotter_does_nothing_when_control_is_zero(
    potential_coefficient, total_evolution_time, number_of_trotter_steps, trotter_qubrick
):
    """Test that the HubbardPlaquetteTrotterization qubrick does nothing when the control is zero."""
    x_dim = 2
    y_dim = 2
    num_state_qubits = 2 * (x_dim * y_dim)

    qc = QPU(num_qubits=num_state_qubits + 1)

    ctrl = Qubits(1, "ctrl", qc)
    psi_reg = Qubits(num_state_qubits, "psi", qc)
    qc.set_param("random_seed", 42)
    qc.set_random()
    ctrl.write(0)
    initial = qc.pull_state()

    fh_data = FermiHubbardData(
        x_dim,
        y_dim,
        total_evolution_time=total_evolution_time,
        n_trotter_steps=number_of_trotter_steps,
        enumeration=None,
        particle_hole_symmetry=True,
        t=1,
        u=potential_coefficient,
    )
    hubbard_time_evolution = trotter_qubrick(InteractionTrotterStep(), PlaquetteTrotterStep(), PlaquetteTrotterStep())

    hubbard_time_evolution.compute(psi_reg, fh_data, ctrl=ctrl)

    final_state = qc.pull_state()

    assert np.allclose(initial, final_state)


@pytest.mark.parametrize("no_trotter_steps", [1, 4, 9])
@pytest.mark.parametrize("lattice_size_x", [4, 6])
def test_merging_trotter_terms(no_trotter_steps, lattice_size_x):
    """Test that the two Hubbard PlaquetteTrotterization qubricks with different Trotter ordering are merging evolutions as expected inbetween successive Trotter steps."""
    t = 1
    u = 8
    lattice_size_x = int(lattice_size_x)
    lattice_size_y = lattice_size_x  # Lattice size dimension along y-axis

    number_spin_sites = 2 * (lattice_size_x * lattice_size_y)
    total_num_qubits = number_spin_sites
    total_num_qubits += 100  # Include enough ancillae

    for trotter_qbrk in [HubbardPlaquetteTrotterizationIPG, HubbardPlaquetteTrotterizationPIG]:
        # Set up QPU instance and Qubits object
        qc = QPU(
            num_qubits=total_num_qubits,
            pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"],
            filters=[">>witness>>", ">>buffer>>"],
        )

        # Declare Registers
        ctrl = DirectionalControlQubit(1, name="ctrl", qpu=qc, defer_rotations=True)
        psi_reg = Qubits(number_spin_sites, "psi", qc)

        # Instantiate data class
        fh_data = FermiHubbardData(
            x_dim=lattice_size_x,
            y_dim=lattice_size_y,
            total_evolution_time=0.32,
            n_trotter_steps=no_trotter_steps,
            enumeration=generate_high_low_enum(lattice_size_x, pink_happy=True),
            u=u,
            t=t,
            particle_hole_symmetry=True,
        )

        # Instantiate sub-qubricks for hubbard time evolution
        hw_qbk = ComputeHammingWeightGroupOfThrees()
        hwp_qbk = PowerOfTwoBatchedDirectionalHammingWeightPhasing(
            hw_qbk,
            n_hwp_batches=1,
            rot_is_rz=True,
            preserve_global_phase=False,
            turn_on_cnots=False,
            use_black_box=False,
        )
        exptXXYY_qbk = exptXXYY(hwp_qbk)

        # Instantiate Plaquette Trotterization Qubrick
        interaction_trotter_step = InteractionTrotterStep(hwp_qbk)
        pink_plaquette_trotter_step = PlaquetteTrotterStep(
            exptXXYY_qubrick=exptXXYY_qbk,
        )
        gold_plaquette_trotter_step = PlaquetteTrotterStep(
            exptXXYY_qubrick=exptXXYY_qbk,
        )

        hubbard_time_evolution = trotter_qbrk(
            interaction_trotter_step,
            pink_plaquette_trotter_step,
            gold_plaquette_trotter_step,
        )

        if trotter_qbrk == HubbardPlaquetteTrotterizationIPG:
            cats = None

            # Reference counts: Trotter order is I P G P I with I merged in between successive trotter steps
            expected_int_count = no_trotter_steps + 1
            expected_plaq_count = 3 * no_trotter_steps

        if trotter_qbrk == HubbardPlaquetteTrotterizationPIG:
            allocate_catalyst_registers_PIG(
                x_dim=lattice_size_x,
                y_dim=lattice_size_y,
                potential_coefficient=u,
                kinetic_coefficient=t,
                total_evolution_time=0.32,
                n_trotter_steps=no_trotter_steps,
                n_hwp_batches=1,
                qc=qc,
            )
            cats = fetch_catalyst_qubits(qc)

            # Reference counts: Trotter order is P I G I P  with P merged in between successive trotter steps
            expected_int_count = 2 * no_trotter_steps
            expected_plaq_count = 2 * no_trotter_steps + 1

        # Compute
        if trotter_qbrk == HubbardPlaquetteTrotterizationIPG:
            hubbard_time_evolution.compute(psi_reg, fh_data, ctrl=ctrl)
        else:
            hubbard_time_evolution.compute(psi_reg, fh_data, catalyst_reg=cats, use_jump_back=True, ctrl=ctrl)

        wb_int_count = qc.witness.filter(name="qc.qbk_compute_start", label="InteractionTrotterStep").count()
        wb_plaq_count = qc.witness.filter(name="qc.qbk_compute_start", label="PlaquetteTrotterStep").count()

        assert wb_int_count == expected_int_count
        assert wb_plaq_count == expected_plaq_count


@pytest.mark.parametrize("lattice_size_x", [4, 6, 8])
def test_no_cat_rots_increases_with_trotter_steps_as_expected(lattice_size_x):
    """Test that the number of catalyst rotations increases as expected for IPG ordering.

    For the ordering IPG, we expect that the number of catalyst rotations increases by one going from a single trotter
    step to multiple trotter steps. This is because the interaction term is merged, creating a new angle for hamming weight phasing
    corresponding to double the evolution time for the interaction term.

    Note this will not be the case for PIG ordering as the gold and pink evolution times are the same (aside from t and t/2 discrepancies)
    so the merged pink angle is equal to the merged gold angle and we expect the number of catalyst to be the same between single and multiple
    trotter implementations.
    """
    t = 1
    u = 8
    n_hwp_batches = 4

    # Using flags define appropiate qubricks
    hamming_weight_qubrick = ComputeHammingWeightGroupOfThrees()
    hwp_qbk = PowerOfTwoBatchedHammingWeightPhasing(hamming_weight_qubrick, n_hwp_batches=n_hwp_batches)
    exptXXYY_qbk = exptXXYY(hwp_qbk)
    fermionic_swap_qubrick = FermionicSwapWithReplaceAVOpt(use_black_box=True)
    two_mode_ffft_qubrick = TwoModeFFFTViaPPRs(use_black_box=True)

    lattice_size_y = lattice_size_x  # Lattice size dimension along y-axis

    number_spin_sites = 2 * (lattice_size_x * lattice_size_y)
    total_num_qubits = number_spin_sites

    total_num_qubits = 100 * lattice_size_x + 20

    for n_trotter_steps in [1, 2, 3]:
        # Set up QPU instance and Qubits object
        qc = QPU(
            num_qubits=total_num_qubits,
            pre_filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"],
            filters=[">>witness>>"],
        )

        # Declare Registers
        ctrl = 0  # NOTE: Here testing the uncontrol version.
        psi_reg = Qubits(number_spin_sites, "psi", qc)

        # Instantiate data class
        fh_data = FermiHubbardData(
            x_dim=lattice_size_x,
            y_dim=lattice_size_y,
            total_evolution_time=0.31,
            n_trotter_steps=n_trotter_steps,
            enumeration=None,
            u=u,
            t=t,
            particle_hole_symmetry=True,
        )

        # Instantiate Plaquette Trotterization Qubrick
        interaction_trotter_step = InteractionTrotterStep(hwp_qbk)
        pink_plaquette_trotter_step = PlaquetteTrotterStep(
            fermionic_swap_qubrick, two_mode_ffft_qubrick, exptXXYY_qubrick=exptXXYY_qbk
        )
        gold_plaquette_trotter_step = PlaquetteTrotterStep(
            fermionic_swap_qubrick, two_mode_ffft_qubrick, exptXXYY_qubrick=exptXXYY_qbk
        )
        hubbard_time_evolution = HubbardPlaquetteTrotterizationIPG(
            interaction_trotter_step, pink_plaquette_trotter_step, gold_plaquette_trotter_step
        )

        # Compute
        hubbard_time_evolution.compute(psi_reg, fh_data, ctrl=ctrl)
        if n_trotter_steps == 1:
            cat_rot_for_one_trotter_steps = len(qc._rotation_catalyst_qubits.keys())
        else:
            cat_rot_for_more_than_one_trotter_steps = len(qc._rotation_catalyst_qubits.keys())

    assert cat_rot_for_more_than_one_trotter_steps == cat_rot_for_one_trotter_steps + 1
