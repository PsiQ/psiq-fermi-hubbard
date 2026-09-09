from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.ops import QPU_op

from psiq_fh.trotterization.fermi_hubbard_data import FermiHubbardData

from ._circuits import CircuitData, baseline_computation, improved_computation
from .utils import exclude_phase_gates


@dataclass
class Variant:
    """Simple dataclass to capture the different settings between `baseline` and
    `improved` workflows.

    Defined to avoid repeated `if variant == ...` guards.

    If other workflows become desireable, this dataclass will need reworked to be
    more easily extensible.

    Args:
        name: The name of this variant. Currently one of `baseline` or `improved`.

    Attributes:
        pink_happy (bool): Whether or not to use the pink_happy setting when generating the
            Fermi Hubbard dataclass.
        rotation_filter_conditions: Extra conditions to include in the rotation filter
            used in the QPU.
        workflow: The workflow function to use. This should modify a QPU in place.

    """

    name: Literal["baseline", "improved"]

    def get_data_filename(self, x_dim: int, y_dim: int, num_batches: int) -> str:
        return f"{x_dim}x{y_dim}_{num_batches}batches_{self.name}.json"

    @property
    def pink_happy(self) -> bool:
        return self.name == "improved"

    @property
    def rotation_filter_conditions(self) -> Callable[[QPU_op], list[bool]] | None:
        if self.name == "baseline":
            return None
        return exclude_phase_gates

    def run_computation(
        self,
        *,
        qc: QPU,
        psi_register: Qubits,
        fermi_hubbard_data: FermiHubbardData,
        circuit_data: CircuitData,
        use_black_box: bool = True,
    ) -> None:
        """Run qubrick computation for the given variant.

        Args:
            qc (QPU): QPU instance.
            psi_register (Qubits): System register
            fermi_hubbard_data (Vanilla2DFermiHubbardData): Data class for the Fermi Hubbard model
            circuit_data (CircuitData): Circuit data from error budget optimization
            use_black_box (bool, optional): Whether to use the black box qubricks for active volume. Defaults to True.
        """
        if self.name == "baseline":
            baseline_computation(
                qc=qc, psi_register=psi_register, fermi_hubbard_data=fermi_hubbard_data, circuit_data=circuit_data
            )
        else:
            improved_computation(
                qc=qc,
                psi_register=psi_register,
                fermi_hubbard_data=fermi_hubbard_data,
                circuit_data=circuit_data,
                use_black_box=use_black_box,
            )
