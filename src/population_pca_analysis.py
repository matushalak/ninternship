from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from src import PLOTSDIR
from src.Analyze import Analyze
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.GLM_new import clean_group_signal
from src.Population import PopulationAnalysis, SessionPCAResult
from src.VisualAreas import Areas


REFERENCE_GROUP = "g2pre"
GROUP_ORDER = ("g1pre", "g1post", "g2pre", "g2post")
GROUP_LABELS = {
    "g1pre": "DR pre",
    "g1post": "DR post",
    "g2pre": "NR pre (reference)",
    "g2post": "NR post",
}

AREA_COLORS = OrderedDict(
    [
        ("V1", "mediumseagreen"),
        ("AM/PM", "coral"),
        ("A/RL/AL", "rosybrown"),
        ("LM", "orchid"),
        ("Unassigned", "lightgrey"),
    ]
)
AREA_ORDER = tuple(AREA_COLORS.keys())[:-1]

NEURON_TYPE_COLORS = OrderedDict(
    [
        ("VIS", "dodgerblue"),
        ("AUD", "red"),
        ("MST", "goldenrod"),
        ("Unclassified", "lightgrey"),
    ]
)

FIT_TRIAL_TYPE_GROUPS = OrderedDict(
    [
        ("V", (6, 7)),
        ("A", (0, 3)),
        ("AV", (1, 2, 4, 5)),
    ]
)


@dataclass
class GroupEmbeddingDataset:
    group: str
    matrix: np.ndarray
    area_labels: np.ndarray
    neuron_type_labels: np.ndarray | None
    session_names: np.ndarray
    feature_groups: np.ndarray
    feature_time: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-post", default="both", choices=("pre", "post", "both"))
    parser.add_argument("--signal-source", default="zsig", choices=("signal", "zsig", "glm_clean"))
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--session-indices", nargs="*", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--plot-dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--pool-by-area", action="store_true")
    parser.add_argument(
        "--fit-trial-types",
        nargs="+",
        default=("V", "A", "AV"),
        choices=tuple(FIT_TRIAL_TYPE_GROUPS),
    )
    parser.add_argument(
        "--fit-time",
        nargs=2,
        type=float,
        metavar=("START_S", "STOP_S"),
        default=(0.0, 1.0),
    )
    parser.add_argument(
        "--plot-time",
        nargs=2,
        type=float,
        metavar=("START_S", "STOP_S"),
        default=(0.0, 1.0),
    )
    parser.add_argument(
        "--color-by",
        nargs="+",
        default=("area",),
        choices=("area", "neuron_type"),
    )
    parser.add_argument(
        "--save-dir",
        default=str(Path(PLOTSDIR) / "population_pca"),
    )
    return parser.parse_args()


def resolve_signal(
    av: AUDVIS,
    signal_source: Literal["signal", "zsig", "glm_clean"],
    pre_post: Literal["pre", "post", "both"],
) -> np.ndarray:
    if signal_source == "glm_clean":
        return clean_group_signal(group_name=av.NAME, pre_post=pre_post)
    if signal_source == "signal":
        return av.signal
    return av.zsig


def get_trial_type_order(av: AUDVIS) -> OrderedDict[str, tuple[int, ...]]:
    return OrderedDict(
        (str(av.int_to_str_trials_map[tt]), (int(tt),))
        for tt in sorted(av.int_to_str_trials_map)
    )


def build_area_labels(av: AUDVIS) -> np.ndarray:
    labels = np.full(av.session_neurons[-1][-1], "Unassigned", dtype=object)
    areas = Areas(av, get_indices=True)
    for area_name, indices in areas.area_indices.items():
        labels[indices] = area_name
    return labels


def build_area_index_map(av: AUDVIS) -> OrderedDict[str, np.ndarray]:
    areas = Areas(av, get_indices=True)
    return OrderedDict((area_name, np.asarray(indices, dtype=int)) for area_name, indices in areas.area_indices.items())


def build_neuron_type_labels(av: AUDVIS) -> np.ndarray:
    labels = np.full(av.session_neurons[-1][-1], "Unclassified", dtype=object)
    analysis = Analyze(av)
    for neuron_type in ("VIS", "AUD", "MST"):
        labels[analysis.NEURON_groups[neuron_type]] = neuron_type
    return labels


def build_group_embedding_dataset(
    av: AUDVIS,
    signal: np.ndarray,
    *,
    session_indices: list[int] | None,
    include_neuron_types: bool,
    fit_time: tuple[float, float],
) -> GroupEmbeddingDataset:
    population = PopulationAnalysis(av, signal=signal, baseline_correct=True)
    trial_order = get_trial_type_order(av)
    fit_time_indices, _ = resolve_plot_indices(
        av,
        population.signal.shape[1],
        plot_time=fit_time,
    )
    global_area_labels = build_area_labels(av)
    global_type_labels = build_neuron_type_labels(av) if include_neuron_types else None

    matrices = []
    area_labels = []
    neuron_type_labels = []
    session_names = []
    feature_groups_ref = None
    feature_time_ref = None

    for session in population.iter_sessions(session_indices=session_indices):
        session_signal, session_neuron_ids = population._select_neurons(session)
        grouped_signal = population._group_session_signal(
            session_signal,
            session.trial_types,
            trial_order,
        )

        blocks = []
        feature_groups = []
        feature_time = []
        for label, tt_group_signal in grouped_signal.items():
            if tt_group_signal.shape[0] == 0:
                raise ValueError(
                    f"{av.NAME} session {session.session_name} had no trials for condition {label}."
                )
            mean_signal = np.nanmean(tt_group_signal[:, fit_time_indices, :], axis=0)
            blocks.append(mean_signal.T)
            feature_groups.extend([label] * mean_signal.shape[0])
            feature_time.extend(fit_time_indices)

        matrix = np.concatenate(blocks, axis=1)
        feature_groups = np.asarray(feature_groups, dtype=object)
        feature_time = np.asarray(feature_time, dtype=int)

        if feature_groups_ref is None:
            feature_groups_ref = feature_groups
            feature_time_ref = feature_time
        else:
            if not (
                np.array_equal(feature_groups_ref, feature_groups)
                and np.array_equal(feature_time_ref, feature_time)
            ):
                raise ValueError(
                    f"Feature alignment changed across sessions in {av.NAME}; cannot pool embeddings."
                )

        matrices.append(matrix)
        area_labels.append(global_area_labels[session_neuron_ids])
        session_names.append(np.full(session_neuron_ids.size, session.session_name, dtype=object))
        if include_neuron_types and global_type_labels is not None:
            neuron_type_labels.append(global_type_labels[session_neuron_ids])

    pooled_matrix = np.vstack(matrices)
    pooled_area_labels = np.concatenate(area_labels)
    pooled_session_names = np.concatenate(session_names)
    pooled_type_labels = (
        np.concatenate(neuron_type_labels) if neuron_type_labels else None
    )

    return GroupEmbeddingDataset(
        group=av.NAME,
        matrix=pooled_matrix,
        area_labels=pooled_area_labels,
        neuron_type_labels=pooled_type_labels,
        session_names=pooled_session_names,
        feature_groups=feature_groups_ref,
        feature_time=feature_time_ref,
    )


def prepare_projection_datasets(
    datasets: OrderedDict[str, GroupEmbeddingDataset],
) -> tuple[OrderedDict[str, GroupEmbeddingDataset], np.ndarray]:
    reference = datasets[REFERENCE_GROUP]
    valid_feature_mask = np.isfinite(reference.matrix).all(axis=0)
    if not np.any(valid_feature_mask):
        raise ValueError("Reference group had no valid features left for pooled embedding PCA.")

    projected = OrderedDict()
    for group_name, dataset in datasets.items():
        if not (
            np.array_equal(reference.feature_groups, dataset.feature_groups)
            and np.array_equal(reference.feature_time, dataset.feature_time)
        ):
            raise ValueError(
                f"Feature order for {group_name} did not match {REFERENCE_GROUP}; cannot project onto the same PCA basis."
            )
        matrix = dataset.matrix[:, valid_feature_mask]
        valid_rows = np.isfinite(matrix).all(axis=1)
        projected[group_name] = GroupEmbeddingDataset(
            group=dataset.group,
            matrix=matrix[valid_rows],
            area_labels=dataset.area_labels[valid_rows],
            neuron_type_labels=None
            if dataset.neuron_type_labels is None
            else dataset.neuron_type_labels[valid_rows],
            session_names=dataset.session_names[valid_rows],
            feature_groups=dataset.feature_groups[valid_feature_mask],
            feature_time=dataset.feature_time[valid_feature_mask],
        )
    return projected, valid_feature_mask


def build_axes_grid(
    nrows: int,
    ncols: int,
    *,
    plot_dim: int,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray]:
    subplot_kw = {"projection": "3d"} if plot_dim == 3 else None
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
        subplot_kw=subplot_kw,
    )
    return fig, axes


def resolve_fit_trial_groups(
    fit_trial_types: tuple[str, ...] | list[str],
) -> OrderedDict[str, tuple[int, ...]]:
    return OrderedDict(
        (label, FIT_TRIAL_TYPE_GROUPS[label])
        for label in fit_trial_types
    )


def fit_trial_types_slug(fit_trial_types: tuple[str, ...] | list[str]) -> str:
    return "-".join(fit_trial_types)


def trial_time_axis(av: AUDVIS, n_timepoints: int) -> np.ndarray:
    pre_sec = float(av.trial_sec[0])
    return (np.arange(n_timepoints, dtype=float) / float(av.SF)) - pre_sec


def resolve_plot_indices(
    av: AUDVIS,
    n_timepoints: int,
    plot_time: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    time_axis = trial_time_axis(av, n_timepoints)
    start_s, stop_s = map(float, plot_time)
    mask = (time_axis >= start_s) & (time_axis <= stop_s)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(
            f"Plot window {plot_time} s selected no samples for {av.NAME}. "
            f"Available time range is {time_axis[0]:.3f} to {time_axis[-1]:.3f} s."
        )
    return indices, time_axis


def resolve_time_indices(
    av: AUDVIS,
    n_timepoints: int,
    time_window: tuple[float, float],
) -> np.ndarray:
    indices, _ = resolve_plot_indices(av, n_timepoints, plot_time=time_window)
    return indices


def scatter_coords(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    palette: OrderedDict[str, str],
    plot_dim: int,
) -> None:
    for label, color in palette.items():
        mask = labels == label
        if not np.any(mask):
            continue
        if plot_dim == 3:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=10,
                alpha=0.7,
                color=color,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.7,
                color=color,
            )


def set_embedding_axis_labels(ax: plt.Axes, plot_dim: int) -> None:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if plot_dim == 3:
        ax.set_zlabel("PC3")


def set_axis_limits(
    ax: plt.Axes,
    mins: np.ndarray,
    maxs: np.ndarray,
    plot_dim: int,
) -> None:
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    if plot_dim == 3:
        ax.set_zlim(mins[2], maxs[2])


def plot_pooled_embeddings(
    datasets: OrderedDict[str, GroupEmbeddingDataset],
    pca: PCA,
    *,
    plot_dim: int,
    color_by: tuple[str, ...],
    save_path: Path,
) -> None:
    projected_coords = OrderedDict(
        (group_name, pca.transform(dataset.matrix)[:, :plot_dim])
        for group_name, dataset in datasets.items()
    )
    all_coords = np.vstack(list(projected_coords.values()))
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    padding = 0.05 * np.maximum(maxs - mins, 1e-8)
    mins -= padding
    maxs += padding

    row_modes = list(color_by)
    col_groups = [group for group in GROUP_ORDER if group in datasets]
    fig, axes = build_axes_grid(
        len(row_modes),
        len(col_groups),
        plot_dim=plot_dim,
        figsize=(5.5 * len(col_groups), 4.8 * len(row_modes)),
    )

    for row_index, mode in enumerate(row_modes):
        palette = AREA_COLORS if mode == "area" else NEURON_TYPE_COLORS
        for col_index, group_name in enumerate(col_groups):
            ax = axes[row_index, col_index]
            dataset = datasets[group_name]
            coords = projected_coords[group_name]
            labels = dataset.area_labels if mode == "area" else dataset.neuron_type_labels
            if labels is None:
                ax.set_axis_off()
                continue
            scatter_coords(ax, coords, labels, palette, plot_dim)
            set_axis_limits(ax, mins, maxs, plot_dim)
            set_embedding_axis_labels(ax, plot_dim)
            ax.set_title(
                f"{GROUP_LABELS.get(group_name, group_name)}\n"
                f"n={coords.shape[0]}"
            )

    legend_handles = []
    for mode in row_modes:
        palette = AREA_COLORS if mode == "area" else NEURON_TYPE_COLORS
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=7,
                    label=f"{mode}: {label}",
                )
                for label, color in palette.items()
            ]
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=8,
    )
    explained = np.sum(pca.explained_variance_ratio_[:plot_dim])
    fig.suptitle(
        "Pooled neuronal embeddings across sessions\n"
        f"Reference PCA fit on {GROUP_LABELS[REFERENCE_GROUP]} | "
        f"variance in first {plot_dim} PCs = {explained:.2f}",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def ensure_components(result: SessionPCAResult, plot_dim: int) -> None:
    if result.scores.shape[1] < plot_dim:
        raise ValueError(
            f"{result.session_name} only produced {result.scores.shape[1]} PCA component(s); "
            f"need at least {plot_dim} for the requested plot."
        )


def add_start_end_markers(
    ax: plt.Axes,
    coords: np.ndarray,
    color: str,
    plot_dim: int,
) -> None:
    start = coords[0]
    end = coords[-1]
    if plot_dim == 3:
        ax.scatter(
            start[0],
            start[1],
            start[2],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=30,
            marker="o",
        )
        ax.scatter(
            end[0],
            end[1],
            end[2],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=42,
            marker="X",
        )
    else:
        ax.scatter(
            start[0],
            start[1],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=30,
            marker="o",
        )
        ax.scatter(
            end[0],
            end[1],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=42,
            marker="X",
        )


def add_trial_boundary_markers(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    color: str,
    onset_index: int | None,
    offset_index: int | None,
    plot_dim: int,
) -> None:
    if onset_index is not None:
        onset = coords[int(onset_index)]
        if plot_dim == 3:
            ax.scatter(
                onset[0],
                onset[1],
                onset[2],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=34,
                marker="s",
            )
        else:
            ax.scatter(
                onset[0],
                onset[1],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=34,
                marker="s",
            )
    if offset_index is not None:
        offset = coords[int(offset_index)]
        if plot_dim == 3:
            ax.scatter(
                offset[0],
                offset[1],
                offset[2],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=38,
                marker="D",
            )
        else:
            ax.scatter(
                offset[0],
                offset[1],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=38,
                marker="D",
            )


def plot_group_manifold_grid(
    population: PopulationAnalysis,
    manifolds: OrderedDict[str, SessionPCAResult],
    *,
    fit_time: tuple[float, float],
    plot_dim: int,
    plot_time: tuple[float, float],
    save_path: Path,
) -> None:
    session_names = list(manifolds)
    n_sessions = len(session_names)
    ncols = math.ceil(math.sqrt(n_sessions))
    nrows = math.ceil(n_sessions / ncols)
    fig, axes = build_axes_grid(
        nrows,
        ncols,
        plot_dim=plot_dim,
        figsize=(5.2 * ncols, 4.8 * nrows),
    )
    axes_flat = axes.ravel()

    plot_indices, time_axis = resolve_plot_indices(
        population.av,
        population.signal.shape[1],
        plot_time=plot_time,
    )

    all_coords = []
    all_trajectories = {}
    for session_name, result in manifolds.items():
        trajectories = population.project_average_trajectories(result)
        trimmed = OrderedDict()
        for trial_type, coords in trajectories.items():
            if coords.shape[1] < plot_dim:
                continue
            coords = coords[plot_indices, :plot_dim]
            trimmed[trial_type] = coords
            all_coords.append(coords)
        all_trajectories[session_name] = trimmed

    if all_coords:
        stacked = np.vstack(all_coords)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        padding = 0.05 * np.maximum(maxs - mins, 1e-8)
        mins -= padding
        maxs += padding
    else:
        mins = maxs = np.zeros(plot_dim)

    ordered_trial_types = [
        str(population.av.int_to_str_trials_map[tt])
        for tt in sorted(population.av.int_to_str_trials_map)
    ]
    trial_type_palette = {
        trial_type: plt.get_cmap("tab10")(idx % 10)
        for idx, trial_type in enumerate(ordered_trial_types)
    }
    trial_type_handles = []
    trial_type_order = []
    onset_candidates = np.where(np.isclose(time_axis[plot_indices], 0.0))[0]
    offset_candidates = np.where(np.isclose(time_axis[plot_indices], 1.0))[0]
    onset_index = int(onset_candidates[0]) if onset_candidates.size else None
    offset_index = int(offset_candidates[-1]) if offset_candidates.size else None

    for ax, session_name in zip(axes_flat, session_names):
        result = manifolds[session_name]
        trajectories = all_trajectories[session_name]
        for trial_type, coords in trajectories.items():
            color = trial_type_palette[trial_type]
            if plot_dim == 3:
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2)
            else:
                ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
            add_start_end_markers(ax, coords, color=color, plot_dim=plot_dim)
            add_trial_boundary_markers(
                ax,
                coords,
                color=color,
                onset_index=onset_index,
                offset_index=offset_index,
                plot_dim=plot_dim,
            )
            if trial_type not in trial_type_order:
                trial_type_order.append(trial_type)
                trial_type_handles.append(
                    Line2D([0], [0], color=color, linewidth=2, label=trial_type)
                )

        set_axis_limits(ax, mins, maxs, plot_dim)
        set_embedding_axis_labels(ax, plot_dim)
        ax.set_title(
            f"{result.session_name}\nvar={np.sum(result.explained_variance_ratio[:plot_dim]):.2f}"
        )

    for ax in axes_flat[n_sessions:]:
        ax.set_axis_off()

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="end",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="stim on",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="stim off",
        ),
    ]
    fig.legend(
        handles=trial_type_handles + marker_handles,
        loc="lower center",
        ncol=min(len(trial_type_handles) + len(marker_handles), 6),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"{population.av.NAME} neural manifolds by session\n"
        f"Fit {fit_time[0]:.2f} to {fit_time[1]:.2f} s; plotted {plot_time[0]:.2f} to {plot_time[1]:.2f} s; "
        "circles mark plotted-window start, X marks plotted-window end",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def build_pooled_group_traces(
    population: PopulationAnalysis,
    *,
    session_indices: list[int] | None,
    neuron_mask: np.ndarray | None = None,
) -> OrderedDict[str, np.ndarray]:
    native_trial_order = get_trial_type_order(population.av)
    pooled_parts = {label: [] for label in native_trial_order}

    for session in population.iter_sessions(session_indices=session_indices):
        session_signal, _ = population._select_neurons(session)
        if neuron_mask is not None:
            session_global_neurons = session.neuron_ids
            local_keep = np.isin(session_global_neurons, neuron_mask)
            if not np.any(local_keep):
                continue
            session_signal = session_signal[:, :, local_keep]
        grouped_signal = population._group_session_signal(
            session_signal,
            session.trial_types,
            native_trial_order,
        )
        for label, tt_signal in grouped_signal.items():
            if tt_signal.shape[0] == 0:
                raise ValueError(
                    f"{population.av.NAME} session {session.session_name} had no trials for condition {label}."
                )
            pooled_parts[label].append(np.nanmean(tt_signal, axis=0))

    return OrderedDict(
        (label, np.concatenate(parts, axis=1))
        for label, parts in pooled_parts.items()
        if parts
    )


def fit_pooled_group_manifold(
    population: PopulationAnalysis,
    *,
    fit_trial_groups: OrderedDict[str, tuple[int, ...]],
    session_indices: list[int] | None,
    n_components: int,
    neuron_mask: np.ndarray | None = None,
    fit_time: tuple[float, float] = (0.0, 1.0),
) -> tuple[PCA, OrderedDict[str, np.ndarray]]:
    native_trial_traces = build_pooled_group_traces(
        population,
        session_indices=session_indices,
        neuron_mask=neuron_mask,
    )
    if not native_trial_traces:
        raise ValueError(f"No pooled traces available for {population.av.NAME}.")
    fit_time_indices = resolve_time_indices(
        population.av,
        population.signal.shape[1],
        time_window=fit_time,
    )

    fit_blocks = []
    for fit_label, component_tts in fit_trial_groups.items():
        native_labels = [population.av.int_to_str_trials_map[tt] for tt in component_tts]
        traces = [native_trial_traces[str(label)] for label in native_labels]
        fit_trace = np.nanmean(np.stack(traces, axis=0), axis=0)
        fit_blocks.append(fit_trace[fit_time_indices])

    fit_matrix = np.vstack(fit_blocks)
    valid_neurons = np.isfinite(fit_matrix).all(axis=0)
    fit_matrix = fit_matrix[:, valid_neurons]
    if fit_matrix.shape[0] < 2 or fit_matrix.shape[1] < 2:
        raise ValueError(
            f"Pooled manifold fit for {population.av.NAME} produced matrix {fit_matrix.shape}, too small for PCA."
        )

    pca = PCA(n_components=min(n_components, *fit_matrix.shape))
    pca.fit(fit_matrix)

    projected = OrderedDict(
        (label, trace[:, valid_neurons])
        for label, trace in native_trial_traces.items()
    )
    return pca, projected


def plot_pooled_group_manifold(
    population: PopulationAnalysis,
    *,
    fit_trial_groups: OrderedDict[str, tuple[int, ...]],
    session_indices: list[int] | None,
    n_components: int,
    fit_time: tuple[float, float],
    plot_dim: int,
    plot_time: tuple[float, float],
    save_path: Path,
    by_area: bool = False,
) -> None:
    plot_indices, time_axis = resolve_plot_indices(
        population.av,
        population.signal.shape[1],
        plot_time=plot_time,
    )
    area_map = build_area_index_map(population.av)
    panel_names = list(AREA_ORDER) if by_area else ["all"]
    fig, axes = build_axes_grid(
        2 if by_area else 1,
        2 if by_area else 1,
        plot_dim=plot_dim,
        figsize=(11, 9) if by_area else (6.5, 5.5),
    )
    axes_flat = axes.ravel()

    projected_by_panel: OrderedDict[str, OrderedDict[str, np.ndarray]] = OrderedDict()
    all_projected = []
    for panel_name in panel_names:
        neuron_mask = area_map.get(panel_name) if by_area else None
        try:
            pca, native_trial_traces = fit_pooled_group_manifold(
                population,
                fit_trial_groups=fit_trial_groups,
                session_indices=session_indices,
                n_components=n_components,
                neuron_mask=neuron_mask,
                fit_time=fit_time,
            )
        except ValueError:
            projected_by_panel[panel_name] = OrderedDict()
            continue
        projected = OrderedDict(
            (label, pca.transform(trace[plot_indices])[:, :plot_dim])
            for label, trace in native_trial_traces.items()
        )
        projected_by_panel[panel_name] = projected
        all_projected.extend(projected.values())

    if not all_projected:
        raise ValueError(f"No pooled manifold trajectories available for {population.av.NAME}.")

    stacked = np.vstack(all_projected)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    padding = 0.05 * np.maximum(maxs - mins, 1e-8)
    mins -= padding
    maxs += padding

    ordered_trial_types = [
        str(population.av.int_to_str_trials_map[tt])
        for tt in sorted(population.av.int_to_str_trials_map)
    ]
    trial_type_palette = {
        trial_type: plt.get_cmap("tab10")(idx % 10)
        for idx, trial_type in enumerate(ordered_trial_types)
    }
    trial_type_handles = []
    onset_candidates = np.where(np.isclose(time_axis[plot_indices], 0.0))[0]
    offset_candidates = np.where(np.isclose(time_axis[plot_indices], 1.0))[0]
    onset_index = int(onset_candidates[0]) if onset_candidates.size else None
    offset_index = int(offset_candidates[-1]) if offset_candidates.size else None

    seen_trial_types = set()
    for ax, panel_name in zip(axes_flat, panel_names):
        projected = projected_by_panel.get(panel_name, OrderedDict())
        if not projected:
            ax.set_axis_off()
            continue
        for trial_type, coords in projected.items():
            color = trial_type_palette[trial_type]
            if plot_dim == 3:
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2)
            else:
                ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
            add_start_end_markers(ax, coords, color=color, plot_dim=plot_dim)
            add_trial_boundary_markers(
                ax,
                coords,
                color=color,
                onset_index=onset_index,
                offset_index=offset_index,
                plot_dim=plot_dim,
            )
            if trial_type not in seen_trial_types:
                seen_trial_types.add(trial_type)
                trial_type_handles.append(
                    Line2D([0], [0], color=color, linewidth=2, label=trial_type)
                )
        set_axis_limits(ax, mins, maxs, plot_dim)
        set_embedding_axis_labels(ax, plot_dim)
        ax.set_title(panel_name if by_area else population.av.NAME)

    for ax in axes_flat[len(panel_names):]:
        ax.set_axis_off()

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="end",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="stim on",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="stim off",
        ),
    ]
    fig.legend(
        handles=trial_type_handles + marker_handles,
        loc="lower center",
        ncol=min(len(trial_type_handles) + len(marker_handles), 6),
        frameon=False,
        fontsize=8,
    )
    fit_slug = fit_trial_types_slug(tuple(fit_trial_groups))
    panel_note = "separate PCA per area pooled across sessions" if by_area else "single PCA pooled across sessions"
    fig.suptitle(
        f"{population.av.NAME} pooled manifold across sessions\n"
        f"{panel_note}; fit on {fit_slug}, PCA fit window {fit_time[0]:.2f} to {fit_time[1]:.2f} s, "
        f"plotted {plot_time[0]:.2f} to {plot_time[1]:.2f} s",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    __main__ = __import__("__main__")
    __main__.Behavior = Behavior

    n_components = max(args.n_components, args.plot_dim)
    color_by = tuple(args.color_by)
    fit_time = tuple(args.fit_time)
    plot_time = tuple(args.plot_time)
    fit_trial_types = tuple(args.fit_trial_types)
    fit_trial_groups = resolve_fit_trial_groups(fit_trial_types)
    fit_slug = fit_trial_types_slug(fit_trial_types)
    save_root = Path(args.save_dir)
    embeddings_save_dir = save_root / "pooled_embeddings" / args.signal_source
    pool_mode = args.pool or args.pool_by_area
    manifold_mode_dir = "pooled_by_area" if args.pool_by_area else ("pooled" if pool_mode else "sessionwise")
    manifolds_save_dir = save_root / "manifolds" / args.signal_source / manifold_mode_dir / f"fit_{fit_slug}"
    embeddings_save_dir.mkdir(parents=True, exist_ok=True)
    manifolds_save_dir.mkdir(parents=True, exist_ok=True)

    loaded = OrderedDict(
        (av.NAME, av)
        for av in load_in_data(pre_post="both")
        if av is not None
    )

    selected_groups = [group for group in GROUP_ORDER if group in loaded]
    if args.pre_post != "both":
        selected_groups = [group for group in selected_groups if group.endswith(args.pre_post)]
    if args.groups is not None:
        requested = set(args.groups)
        selected_groups = [group for group in selected_groups if group in requested]

    if REFERENCE_GROUP not in selected_groups:
        raise ValueError(
            f"{REFERENCE_GROUP} must be included because pooled embeddings are anchored on that reference PCA."
        )

    embedding_datasets: OrderedDict[str, GroupEmbeddingDataset] = OrderedDict()
    populations: OrderedDict[str, PopulationAnalysis] = OrderedDict()
    manifold_results: OrderedDict[str, OrderedDict[str, SessionPCAResult]] = OrderedDict()

    for group_name in selected_groups:
        av = loaded[group_name]
        signal = resolve_signal(
            av,
            signal_source=args.signal_source,
            pre_post="post" if "post" in av.NAME else "pre",
        )

        embedding_datasets[group_name] = build_group_embedding_dataset(
            av,
            signal,
            session_indices=args.session_indices,
            include_neuron_types="neuron_type" in color_by,
            fit_time=fit_time,
        )

        population = PopulationAnalysis(av, signal=signal, baseline_correct=True)
        populations[group_name] = population
        if not pool_mode:
            manifolds = population.neural_manifold(
                trial_groups=fit_trial_groups,
                session_indices=args.session_indices,
                time_window=resolve_time_indices(
                    av,
                    population.signal.shape[1],
                    time_window=fit_time,
                ),
                n_components=n_components,
            )
            for result in manifolds.values():
                ensure_components(result, args.plot_dim)
            manifold_results[group_name] = manifolds

    projected_datasets, _ = prepare_projection_datasets(embedding_datasets)
    reference_dataset = projected_datasets[REFERENCE_GROUP]
    reference_pca = PCA(n_components=min(n_components, *reference_dataset.matrix.shape))
    reference_pca.fit(reference_dataset.matrix)

    pooled_name = f"pooled_neuronal_embeddings_{args.signal_source}_{args.plot_dim}d.png"
    plot_pooled_embeddings(
        projected_datasets,
        reference_pca,
        plot_dim=args.plot_dim,
        color_by=color_by,
        save_path=embeddings_save_dir / pooled_name,
    )

    for group_name in selected_groups:
        if pool_mode:
            manifold_name = (
                f"{group_name}_neural_manifold_"
                f"{'pooled_by_area' if args.pool_by_area else 'pooled'}_"
                f"{args.signal_source}_{args.plot_dim}d.png"
            )
            plot_pooled_group_manifold(
                populations[group_name],
                fit_trial_groups=fit_trial_groups,
                session_indices=args.session_indices,
                n_components=n_components,
                fit_time=fit_time,
                plot_dim=args.plot_dim,
                plot_time=plot_time,
                save_path=manifolds_save_dir / manifold_name,
                by_area=args.pool_by_area,
            )
        else:
            manifold_name = f"{group_name}_neural_manifold_sessions_{args.signal_source}_{args.plot_dim}d.png"
            plot_group_manifold_grid(
                populations[group_name],
                manifold_results[group_name],
                fit_time=fit_time,
                plot_dim=args.plot_dim,
                plot_time=plot_time,
                save_path=manifolds_save_dir / manifold_name,
            )
        print(f"Saved {group_name} manifold plot to {manifolds_save_dir / manifold_name}")

    print(f"Manifold fit trial types: {fit_slug}")
    print(f"Saved pooled embeddings to {embeddings_save_dir / pooled_name}")


if __name__ == "__main__":
    main()
