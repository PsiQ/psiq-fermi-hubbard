"""Utility functions for the Fermi-Hubbard Trotter Workbench program."""

from pathlib import Path

import psiqdk.workbench.opcodes as opc
from psiqdk.workbench.ops import QPU_op
from psiqdk.workbench.ops._check_ops import is_phase

import numpy as np
import json


def project_root() -> Path:
    """Determine the project root.

    This function assumes a local clone, as it hunts for a pyproject.toml,
    and will not work if called from a PyPI installation.
    """
    start = Path(__file__)
    for path in start.resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate `pyproject.toml` to determine path to project root.")


def exclude_phase_gates(op: QPU_op) -> list[bool]:
    """Determine if an operation is any arbitrary Z-type rotation.

    We do this as we don't want PPRs to be decomposed and treated with mixed
    fallback.

    Note: psiqdk.workbench.ops._check_ops.is_phase explicitly excludes arbitrary Z-type
    rotations, and includes only fixed angle rotations like Z, S, T and inverses.
    """
    return [is_phase(op.opcode) or (op.opcode == opc.OP_qc_rz)]


def compute_T_rotation_synthesis_mixed_fallback(epsilon: float = 1e-6) -> float:
    """Compute average T-gate count for mixed-fallback rotation synthesis.

    Args:
        epsilon: Error per rotation.

    Returns:
        Average T-gate count as a float.
    """
    return 0.53 * np.log2(1 / epsilon) + 4.86


def extract_all(metrics: dict[int, dict[str, int | float]], key: str) -> np.array:
    """Extract all values for a given key from the metrics data.

    Args:
        metrics (dict[int, dict[str, int | float]]): Metrics data.
        key (str): Key to extract.
    """
    return np.array([x[key] for x in metrics.values()])


def json_dump(data: dict[str, ...] | list[...], filename: str, out_dir: Path):
    """Dump data to a JSON file.

    Args:
        data (dict[str, ...] | list[...]): Data to dump.
        filename (str): Filename to dump to.
        out_dir (Path): Directory to dump to.
    """
    with open(out_dir / f"{filename}.json", "w") as f:
        json.dump(data, f)
