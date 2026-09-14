"""Quick and dirty code to process the data contained in `.qre_analysis` files
and generate nice resource breakdown diagrams.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from psiq_fh.wb_program import QreAnalysis


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
) -> Axes:
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
