"""Trotterization qubricks for the 2D Fermi-Hubbard model.

These are the names used in the accompanying paper. Everything else stays reachable by its
module path.
"""

from .catalyst_allocation import allocate_catalyst_registers_IPG, allocate_catalyst_registers_PIG
from .directional_hamming_weight_phasing import (
    DirectionalHammingWeightPhasing,
    PowerOfTwoBatchedDirectionalHammingWeightPhasing,
)
from .fermi_hubbard_data import InteractionTermData, PlaquetteTermData, FermiHubbardData
from .fermi_hubbard_trotterization import (
    HubbardPlaquetteTrotterizationIPG,
    HubbardPlaquetteTrotterizationPIG,
    HubbardPlaquetteTrotterizationPIGClosedControl,
)
from .fswapping.fermionic_swap import FermionicSwapWithReplace, FermionicSwapWithReplaceAVOpt
from .fswapping.fswap_network import NaivefSWAPNetworkWithReplace, PinkLocalizedFermionicSwapNetworkWithReplace
from .hopping import PlaquetteTrotterStep, TwoModeFFFTViaControlledHad, TwoModeFFFTViaPPRs, exptXXYY, exptXXYYViaPPR
from .interaction import InteractionTrotterStep

__all__ = [
    "FermiHubbardData",
    "InteractionTermData",
    "PlaquetteTermData",
    "HubbardPlaquetteTrotterizationIPG",
    "HubbardPlaquetteTrotterizationPIG",
    "HubbardPlaquetteTrotterizationPIGClosedControl",
    "InteractionTrotterStep",
    "PlaquetteTrotterStep",
    "exptXXYY",
    "exptXXYYViaPPR",
    "TwoModeFFFTViaControlledHad",
    "TwoModeFFFTViaPPRs",
    "NaivefSWAPNetworkWithReplace",
    "PinkLocalizedFermionicSwapNetworkWithReplace",
    "FermionicSwapWithReplace",
    "FermionicSwapWithReplaceAVOpt",
    "DirectionalHammingWeightPhasing",
    "PowerOfTwoBatchedDirectionalHammingWeightPhasing",
    "allocate_catalyst_registers_IPG",
    "allocate_catalyst_registers_PIG",
]
