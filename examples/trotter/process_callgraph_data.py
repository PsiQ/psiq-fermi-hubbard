"""Quick and dirty code to process the data contained in `.qre_analysis` files
and generate nice resource breakdown diagrams.
"""

from __future__ import annotations

import re
import json
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from typing_extensions import Literal


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


def group_active_volume_components(
    components: Mapping[str, float],
    *,
    groups: dict[str, tuple[str, ...]],
    others_label: str = "Other",
) -> dict[str, float]:
    """Group active-volume components by substrings in their names.

    Each component is assigned to the first matching group. Components that
    match no group are combined under ``others_label``.
    """
    grouped = OrderedDict((label, 0.0) for label in groups)
    grouped[others_label] = 0.0

    for component_name, value in components.items():
        for label, substrings in groups.items():
            if any(substring in component_name for substring in substrings):
                grouped[label] += float(value)
                break
        else:
            grouped[others_label] += float(value)

    # Remove empty groups while preserving the order.
    return {label: value for label, value in grouped.items() if value != 0}


def plot_av_breakdown_over_lattice_sizes(
    data: list[QreAnalysis],
    lattice_sizes: list[int],
    *,
    metric="active_volume",
    granularity=("PhasingCircuit",),
    groups=None,
    others_label="Other",
    title="Active volume breakdown",
    ylabel="Active volume",
    legend_top_to_bottom=True,
    show_percentages=False,
    ax=None,
    figsize=(8, 6),
    ylim_high=None,
    show_legend=True,
    show_xlabel=True,
    show_ylabel=True,
)->Axes:
    """Plot a stacked active-volume breakdown for multiple call graphs.

    Args:
        data: Sequence of `QreAnaylsis` classes, one per lattice size.
        lattice_sizes: Sequence of lattice sizes corresponding to ``data``.
        metric: Metric to extract.
        granularity: The Qubrick granularity to stop at.
        groups: Mapping from displayed category to component-name substrings.
        others_label: Category for unmatched components.
        show_percentages: If True, place percentages inside sufficiently large stack sections.

    Returns:
        matplotlib.axes.Axes
    """
    if groups is None:
        groups = OrderedDict(
            [
                ("FermionicSwap", ("FermionicSwap",)),
                (
                    "ComputeHammingWeight",
                    ("ComputeHammingWeight",),
                ),
                ("PhasingCircuit", ("PhasingCircuit",)),
                ("TwoModeFFFT", ("TwoModeFFFT",)),
            ]
        )

    if len(data) != len(lattice_sizes):
        raise ValueError("graphs and lattice_sizes must have the same length.")

    category_names = [*groups, others_label]
    breakdowns = []
    expected_totals = []

    for qre in data:
        components = qre.decompose_metric_cost(metric, granularity)
        grouped = group_active_volume_components(
            components,
            groups=groups,
            others_label=others_label,
        )

        # Ensure every graph has the same category keys.
        grouped = {category: float(grouped.get(category, 0.0)) for category in category_names}
        breakdowns.append(grouped)
        expected_totals.append(sum(grouped.values()))

    x = np.arange(len(lattice_sizes))
    bottoms = np.zeros(len(lattice_sizes), dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("Set2")

    handles = []
    labels = []

    for index, category in enumerate(category_names):
        values = np.array(
            [breakdown[category] for breakdown in breakdowns],
            dtype=float,
        )

        bars = ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.7,
            label=category,
            color=cmap(index % cmap.N),
            edgecolor="black",
            linewidth=0.7,
        )

        handles.append(bars[0])
        labels.append(category)

        if show_percentages:
            totals = np.asarray(expected_totals)

            for bar, value, bottom, total in zip(
                bars,
                values,
                bottoms,
                totals,
            ):
                percentage = 100 * value / total

                # Avoid unreadable labels in very small sections.
                if percentage >= 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=14,
                        rotation=90,
                    )

        bottoms += values

    if legend_top_to_bottom:
        handles = handles[::-1]
        labels = labels[::-1]

    if show_legend:
        ax.legend(
            handles,
            labels,
            title="Subroutine",
            frameon=True,
        )

    if ylim_high:
        ax.set_ylim(0, ylim_high)

    ax.set_xticks(x)
    ax.set_xticklabels(lattice_sizes)
    if show_xlabel:
        ax.set_xlabel("Lattice size")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=24)

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )

    ax.grid(axis="y", alpha=1, linestyle=":")
    ax.set_axisbelow(True)
    ax.figure.tight_layout()

    return ax
