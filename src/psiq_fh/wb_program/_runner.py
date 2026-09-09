from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable
from typing import Literal

from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.compilation.filters.elementary_filters._ross_selinger_synthesis import (
    RossSelingerSynthFilter,
)
from psiqdk.workbench.compilation.filters.elementary_filters._single_control import SingleControlFilter
from psiqdk.workbench.integrations import resource_analyzer
from psiqdk.workbench.qre import resource_estimator
from psiqdk.workbench.rotations._rotation import RotationViaMixedFallback

from psiq_fh.trotterization.catalyst_allocation import (
    fetch_catalyst_qubits,
)
from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData
from psiq_fh.utils.jw_ordering_utils import generate_high_low_enum

warnings.filterwarnings("ignore")

from ._circuits import CircuitData
from ._variants import Variant
from .utils import project_root

DATA_PATHWAY = project_root() / "data" / "trotter" / "error_budgeting_data"
"""Absolute pathway to the data directory on the project root."""

PLOT_PATHWAY = project_root() / "examples" / "trotter" / "paper_data_and_plots"
"""Absolute pathway to the trotter plotting folder."""


class FermiHubbardQRERunner:
    """Runner class for the Fermi-Hubbard Trotter Workbench program for QREs."""

    FH_POTENTIAL: float = 8
    FH_KINETIC: float = 1

    def __init__(
        self,
        lattice_sizes: Iterable[int],
        variant: Literal["baseline", "improved"],
        num_batches: int = 1,
        **overrides: int | float,
    ):
        self.lattice_sizes = lattice_sizes
        self.variant = Variant(variant)
        self.num_batches = num_batches
        self.overrides = overrides
        self._validate_overrides()

        self._plotting_subdir = f"{variant}_callgraph_data/{self.num_batches}batches"

    def _validate_overrides(self):
        OVERRIDABLE: set[str] = {"no_trotter_steps", "no_queries"}
        if cant_override := self.overrides.keys() - OVERRIDABLE:
            raise ValueError(
                f"Cannot override the following fields: {cant_override}. Permitted override fields are {OVERRIDABLE}."
            )

    def _new_qc(
        self,
        use_mixed_fallback: bool = True,
        use_black_box: bool = True,
        rotation_filter_conditions: Callable[[...], list[bool]] | None = None,
    ) -> QPU:
        """Returns a fresh QPU  with the following pre-filters:
            - `clean-ladder-filter`
            - `SingleControlFilter(use_anc=True)`
            - `RossSelingerSynthFilter`
                - The `.synth_qbk` attribure of this is set to `RotatedViaMixedFallback`
                    if `use_mixed_fallback=True`
                - If any supplementary rotation_filter_conditions are provided, those
                    are set by the `.extra_conds` attribute.
            - `witness`
        and the `buffer` filter.

        Returns:
            QPU: a fresh QPU instance.
        """
        rotation_filter = RossSelingerSynthFilter()
        if use_mixed_fallback:
            rotation_filter.synth_qbk = RotationViaMixedFallback(use_black_box=use_black_box)
        if rotation_filter_conditions is not None:
            rotation_filter.extra_conds = rotation_filter_conditions
        single_control_filter = SingleControlFilter(use_anc=True)
        return QPU(
            pre_filters=[">>clean-ladder-filter>>", single_control_filter, rotation_filter, ">>witness>>"],
            filters=[">>buffer>>"],
        )

    def load_circuit_data(self, x_dim: int, y_dim: int, raw: bool = False) -> CircuitData:
        circuit_data_file = self.variant.get_data_filename(x_dim, y_dim, num_batches=self.num_batches)
        with open(DATA_PATHWAY / circuit_data_file, "r") as f:
            _raw_data = json.load(f)
        if raw:
            return CircuitData(**_raw_data)
        return CircuitData(**(_raw_data | self.overrides | {"no_batches": self.num_batches}))

    def run_computation(
        self,
        x_dim: int,
        y_dim: int,
        use_mixed_fallback: bool = True,
        use_black_box: bool = True,
        save_callgraph: bool = False,
        total_num_qubits: int | None = None,
    ) -> tuple[QPU, dict[str, int | float]]:
        """Args:
            x_dim: The x dimension of the physical lattice in the target system.
            y_dim: The y dimension of the physical lattice in the target system.
            use_mixed_fallback: If True, uses the mixed fallback for rotation synthesis. Defaults to True.
            use_black_box: If True, uses black box qubricks for active volume. Defaults to True.
            save_callgraph: If True, saves the callgraph data. Defaults to False.
            total_num_qubits: If None, allocates enough qubits to fit the circuit. Defaults to None.

        Returns:
            tuple[QPU, dict[str, int | float]]
        """
        circuit_data = self.load_circuit_data(x_dim, y_dim, raw=False)
        number_spin_sites = 2 * x_dim * y_dim

        # Allocating enough qubits to fit the circuit
        if total_num_qubits is None:
            total_num_qubits = 100 * x_dim + 20

        fermi_hubbard_data = FermiHubbardData(
            x_dim=x_dim,
            y_dim=y_dim,
            total_evolution_time=circuit_data.evolution_time,
            n_trotter_steps=circuit_data.no_trotter_steps,
            enumeration=generate_high_low_enum(x_dim, pink_happy=self.variant.pink_happy),
            u=self.FH_POTENTIAL,
            t=self.FH_KINETIC,
            particle_hole_symmetry=True,
        )

        # Rotation errors (catalyst rotations and non-catalyst rotations)
        eps_cats = 2**-circuit_data.cat_bop
        eps_noncats = 2**-circuit_data.rot_bop

        qc = self._new_qc(
            use_mixed_fallback=use_mixed_fallback,
            use_black_box=use_black_box,
            rotation_filter_conditions=self.variant.rotation_filter_conditions,
        )
        qc.reset(total_num_qubits)

        psi_reg = Qubits(number_spin_sites, "psi", qc)

        with qc.override_rotation_epsilon(eps_noncats):
            with qc.override_catalyst_rotation_epsilon(eps_cats):
                self.variant.run_computation(
                    qc=qc,
                    psi_register=psi_reg,
                    fermi_hubbard_data=fermi_hubbard_data,
                    circuit_data=circuit_data,
                    use_black_box=use_black_box,
                )

        cats: Qubits = fetch_catalyst_qubits(qc)
        metrics = resource_estimator(qc).resources(expanded=True)
        metrics["n_catalyst_rots"] = cats.num_qubits

        if save_callgraph:
            filename_format = f"{self.variant.name}_{x_dim}dim_{self.num_batches}batches.qre_analysis"
            resource_analyzer.export(qc, PLOT_PATHWAY / self._plotting_subdir / filename_format)

        return qc, metrics

    def retrieve(
        self, use_mixed_fallback: bool = True, use_black_box: bool = True, save_callgraphs: bool = False
    ) -> dict[int, dict[str, int | float]]:
        """Retrieve the metrics data for the given lattice sizes.

        Args:
            use_mixed_fallback: Whether to use the mixed fallback for rotation synthesis.
                Defaults to True.
            use_black_box: Whether to use the black box qubricks for active volume. Defaults to True.
            save_callgraphs: Whether to save the callgraph data. Defaults to False.

        Returns:
            dict[int, dict[str, int | float]]: metrics data for the given lattice sizes.
        """
        return {
            lattice: self.run_computation(
                lattice,
                lattice,
                use_mixed_fallback=use_mixed_fallback,
                use_black_box=use_black_box,
                save_callgraph=save_callgraphs,
            )[1]
            for lattice in self.lattice_sizes
        }
