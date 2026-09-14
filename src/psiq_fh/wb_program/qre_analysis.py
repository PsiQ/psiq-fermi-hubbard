"""Models and helpers for processing QRE analysis callgraphs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QreAnalysis:
    """The JSON structure of a .qre_analysis file.

    This dataclass is a result of the creation of all `.qre_analysis` files not being
    open sourced. Typically we would not need to do this kind of post-processing on
    `.qre_analysis` files.

    The `nodes` field of a `.qre_analysis` file represents a flattened tree of Qubricks and their corresponding
    metric costs. As such, rather than reconstructing the tree and traversing it we can simply
    iterate through each element as required.
    """

    record_type: Literal["psiquantum.qre-analysis"]
    title: str
    nodes: list[QreAnalysisNode]

    @classmethod
    def from_json(cls, filepath: str) -> QreAnalysis:
        """Directly open a `.qre_analysis` file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            record_type=data["record_type"],
            title=data["title"],
            nodes=[QreAnalysisNode(**node) for node in data["nodes"]],
        )

    def decompose_metric_cost(self, metric: str, qubrick_granularity: tuple[str, ...] = ()) -> Mapping[str, float]:
        """Decompose the cost of a metric into its per-Qubrick contributions, with optional granularity.

        The returned dictionary is indexed by normalized Qubrick name, such that, for example,
        `<my_qubrick>_compute_0` and `<my_qubrick>_uncompute_2` would have their values summed.

        `qubrick_granularity` should be a tuple of normalized node names in `nodes`.

        If `qubrick_granularity` is unset, each unique Qubrick in the QRE will have an entry in the resulting dictionary.

        If `qubrick_granularity` is set, any `node` that has that Qubrick name in its parentage is skipped.
        The node whose name matches an entry in `qubrick_granularity` exactly contributes its total metric cost, such
        that its children are also included in the costing.

        Args:
            metric: Which metric to decompose.
            qubrick_granularity: A qubrick name to stop traversing the tree at, by default None.

        Returns:
            dict[str, float]: A dictionary of qubrick_name: metric cost. Zero values are dropped.
        """
        decomposition = defaultdict(int)
        for node in self.nodes:
            if any(qbk_name in (node.parent or "") for qbk_name in qubrick_granularity):
                continue
            if node.name in qubrick_granularity:
                decomposition[node.name] += node.total_resource_cost(metric)
            else:
                decomposition[node.name] += node.self_resource_cost(metric)
        return {key: val for key, val in decomposition.items() if val}


@dataclass
class QreAnalysisNode:
    """The JSON structure of a single node in the `nodes` entry
    of QreAnalysis.

    This dataclass is a result of the creation of all `.qre_analysis` files not being
    open sourced. Typically we would not need to do this kind of post-processing on
    `.qre_analysis` files.
    """

    id: str
    name: str
    parent: str | None
    type: str | None
    metrics: dict[str, dict[str, float]]
    meta: None
    children: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Normalize the node name after initialization."""
        self.name = strip_compute_uncompute(self.name)

    def self_resource_cost(self, resource: str) -> float:
        """Return the resource cost that this Node contributes by itself,
        excluding children.
        """
        return self.metrics[resource]["self"]

    def total_resource_cost(self, resource: str) -> float:
        """Return the resource cost that this Node contributes, including all of its
        children.
        """
        return self.metrics[resource]["total"]


def strip_compute_uncompute(node_name: str) -> str:
    """During computation, Qubrick names are appended with `_compute_xx`
    and `_uncompute_xx` to decipher which stage and order they occur in.

    This function strips those suffixes and returns only the bare Qubrick name,
    for easier comparison.
    """
    return re.sub(r"_(?:un)?compute_\d+$", "", node_name)
