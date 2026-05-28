from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import PLOTSDIR


GROUP_ORDER = ["g1pre", "g2pre", "g1post", "g2post"]
AREA_ORDER = ["V1", "AM/PM", "A/RL/AL", "LM"]
AVERAGED_TASK_ORDER = [
    "visual_direction",
    "auditory_direction",
    "congruency",
    "modality",
    "trial_type",
]
POSITION_TASK_ORDER = ["V_position", "A_position"]
GROUP_PALETTE = {
    "g1pre": "#4C78A8",
    "g2pre": "#E45756",
    "g1post": "#72B7B2",
    "g2post": "#F58518",
}
BALANCED_PALETTE = {False: "#c7c7c7", True: "#111111"}
TASK_CHANCE_LEVELS = {
    "visual_direction": 0.5,
    "auditory_direction": 0.5,
    "congruency": 0.5,
    "modality": 1 / 3,
    "trial_type": 1 / 8,
    "V_position": 1 / 15,
    "A_position": 1 / 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="pydata/decoding_population_sessionwise/pre_glm_clean",
    )
    parser.add_argument(
        "--save-dir",
        default=str(Path(PLOTSDIR) / "decoding_population_sessionwise" / "pre_glm_clean"),
    )
    parser.add_argument("--balanced-min-sessions", type=int, default=1)
    parser.add_argument("--balanced-min-neurons", type=int, default=40)
    parser.add_argument("--skip-balanced", action="store_true")
    return parser.parse_args()


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_outputs(
    input_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
    return {
        "averaged": pd.read_csv(input_dir / "session_population_averaged.csv"),
        "position": pd.read_csv(input_dir / "session_population_position.csv"),
        "temporal": _read_optional_csv(input_dir / "session_population_temporal.csv"),
        "counts": _read_optional_csv(input_dir / "session_area_neuron_counts.csv"),
        "balanced_averaged": _read_optional_csv(input_dir / "balanced_session_population_averaged.csv"),
        "balanced_position": _read_optional_csv(input_dir / "balanced_session_population_position.csv"),
        "balanced_temporal": _read_optional_csv(input_dir / "balanced_session_population_temporal.csv"),
    }


def unique_neuron_counts(
    averaged: pd.DataFrame,
    position: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["group", "session_index", "session_name", "area", "n_neurons", "signal_source"]
    counts = pd.concat([averaged[cols], position[cols]], ignore_index=True)
    counts = counts.drop_duplicates()
    return counts.sort_values(["group", "area", "n_neurons"], ascending=[True, True, False])


def compute_balanced_thresholds(
    counts: pd.DataFrame,
    min_sessions_per_group: int = 2,
    min_neurons_per_session: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    inclusion_frames = []

    for area, area_df in counts.groupby("area", sort=False):
        groups = sorted(area_df["group"].unique())
        per_group_counts = {
            group: sorted(
                area_df.loc[area_df["group"] == group, "n_neurons"].tolist(),
                reverse=True,
            )
            for group in groups
        }

        eligible_counts = {
            group: [val for val in vals if val >= min_neurons_per_session]
            for group, vals in per_group_counts.items()
        }

        if any(len(vals) < min_sessions_per_group for vals in eligible_counts.values()):
            threshold = np.nan
            raw_threshold = np.nan
            threshold_reason = "too_few_sessions_after_neuron_filter"
        else:
            raw_threshold = min(val for vals in eligible_counts.values() for val in vals)
            threshold = raw_threshold
            threshold_reason = "ok"

        summary = {
            "area": area,
            "balanced_min_sessions_per_group": min_sessions_per_group,
            "balanced_min_neurons_per_session": min_neurons_per_session,
            "balanced_raw_neuron_threshold": raw_threshold,
            "balanced_neuron_threshold": threshold,
            "balanced_threshold_reason": threshold_reason,
        }

        for group in groups:
            group_mask = area_df["group"] == group
            if np.isnan(threshold):
                included = pd.Series(False, index=area_df.index[group_mask])
            else:
                included = area_df.loc[group_mask, "n_neurons"] >= threshold
            group_rows = area_df.loc[group_mask].copy()
            group_rows["balanced_included"] = included.to_numpy()
            inclusion_frames.append(group_rows)

            summary[f"{group}_sessions_total"] = int(group_mask.sum())
            summary[f"{group}_sessions_balanced"] = int(included.sum())
            summary[f"{group}_neurons_max"] = int(area_df.loc[group_mask, "n_neurons"].max())
            summary[f"{group}_neurons_min"] = int(area_df.loc[group_mask, "n_neurons"].min())

        threshold_rows.append(summary)

    thresholds = pd.DataFrame(threshold_rows)
    inclusion = pd.concat(inclusion_frames, ignore_index=True)
    inclusion["balanced_included"] = inclusion["balanced_included"].astype(bool)
    return thresholds, inclusion


def compute_decoding_inclusion(
    counts: pd.DataFrame,
    min_neurons_per_session: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inclusion = counts.copy()
    if "decoding_included" in inclusion.columns:
        inclusion["balanced_included"] = inclusion["decoding_included"].astype(bool)
    else:
        inclusion["balanced_included"] = inclusion["n_neurons"] >= min_neurons_per_session

    thresholds = pd.DataFrame(
        [
            {
                "area": area,
                "balanced_min_sessions_per_group": np.nan,
                "balanced_min_neurons_per_session": min_neurons_per_session,
                "balanced_raw_neuron_threshold": min_neurons_per_session,
                "balanced_neuron_threshold": min_neurons_per_session,
                "balanced_threshold_reason": "decoding_min_neurons",
            }
            for area in _ordered_areas(inclusion["area"])
        ]
    )
    return thresholds, inclusion


def summarise_balanced_performance(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(
            ["group", "area", "task", "analysis", "signal_source", "scoring"],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            sem_score=("score", "sem"),
            n_sessions=("session_name", "nunique"),
            mean_neurons=("n_neurons", "mean"),
            min_neurons=("n_neurons", "min"),
            max_neurons=("n_neurons", "max"),
        )
    )


def _ordered_groups(values: pd.Series) -> list[str]:
    return [group for group in GROUP_ORDER if group in values.unique()]


def _ordered_areas(values: pd.Series) -> list[str]:
    return [area for area in AREA_ORDER if area in values.unique()]


def plot_performance(
    frame: pd.DataFrame,
    task_order: list[str],
    save_path: str | Path,
    title: str,
) -> None:
    if frame.empty:
        return

    data = frame.copy()
    data["group"] = pd.Categorical(data["group"], categories=_ordered_groups(data["group"]), ordered=True)
    data["area"] = pd.Categorical(data["area"], categories=_ordered_areas(data["area"]), ordered=True)
    data["task"] = pd.Categorical(data["task"], categories=[task for task in task_order if task in data["task"].unique()], ordered=True)

    sns.set_theme(style="whitegrid", context="talk")
    tasks = [task for task in task_order if task in data["task"].unique()]
    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        sharey=False,
        squeeze=False,
    )

    for ax, task in zip(axes.flat, tasks):
        task_df = data.loc[data["task"] == task].copy()
        sns.stripplot(
            data=task_df,
            x="area",
            y="score",
            hue="group",
            dodge=True,
            jitter=0.18,
            alpha=0.55,
            size=6,
            palette=GROUP_PALETTE,
            ax=ax,
        )
        sns.pointplot(
            data=task_df,
            x="area",
            y="score",
            hue="group",
            dodge=0.4,
            errorbar="se",
            markers="D",
            linestyles="-",
            palette=GROUP_PALETTE,
            ax=ax,
        )
        chance_level = TASK_CHANCE_LEVELS.get(task)
        if chance_level is not None:
            ax.axhline(
                chance_level,
                color="#444444",
                linestyle="--",
                linewidth=1.4,
                zorder=1,
            )
            xmin, xmax = ax.get_xlim()
            ax.text(
                xmax,
                chance_level,
                f" chance = {chance_level:.3f}",
                ha="right",
                va="bottom",
                fontsize=10,
                color="#444444",
            )
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, chance_level - 0.03), ymax)
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylabel("Balanced accuracy")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        unique = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique.append((handle, label))
        fig.legend(
            [item[0] for item in unique],
            [item[1] for item in unique],
            loc="upper center",
            ncol=min(4, len(unique)),
            frameon=False,
        )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_neuron_count_diagnostic(
    counts: pd.DataFrame,
    thresholds: pd.DataFrame,
    save_path: str | Path,
) -> None:
    if counts.empty:
        return

    data = counts.copy()
    data["group"] = pd.Categorical(data["group"], categories=_ordered_groups(data["group"]), ordered=True)
    areas = _ordered_areas(data["area"])

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(areas),
        figsize=(5.0 * len(areas), 5.6),
        sharey=False,
        squeeze=False,
    )

    for ax, area in zip(axes.flat, areas):
        area_df = data.loc[data["area"] == area].copy()
        threshold_row = thresholds.loc[thresholds["area"] == area]
        threshold = threshold_row["balanced_neuron_threshold"].iloc[0] if not threshold_row.empty else np.nan
        groups = _ordered_groups(area_df["group"])

        for group_index, group in enumerate(groups):
            group_df = area_df.loc[area_df["group"] == group].copy()
            if group_df.empty:
                continue
            jitter = np.linspace(-0.16, 0.16, num=len(group_df)) if len(group_df) > 1 else np.array([0.0])
            x = np.full(len(group_df), group_index, dtype=float) + jitter
            group_color = GROUP_PALETTE.get(group, "#666666")
            included_mask = group_df["balanced_included"].to_numpy(dtype=bool)

            ax.scatter(
                x[~included_mask],
                group_df.loc[~included_mask, "n_neurons"],
                s=62,
                facecolors="none",
                edgecolors=group_color,
                linewidths=1.6,
                zorder=2,
            )
            ax.scatter(
                x[included_mask],
                group_df.loc[included_mask, "n_neurons"],
                s=92,
                c=group_color,
                alpha=0.95,
                edgecolors="#111111",
                linewidths=1.4,
                zorder=3,
            )

        if np.isfinite(threshold):
            ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.5)
            ax.text(
                0.02,
                threshold,
                f" min N = {int(threshold)}",
                ha="left",
                va="bottom",
                fontsize=10,
                color="#222222",
            )

        ax.set_title(area)
        ax.set_xlabel("")
        ax.set_ylabel("Neurons per session")
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=20)

    group_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=group, markersize=9)
        for group, color in GROUP_PALETTE.items()
        if group in data["group"].unique()
    ]
    inclusion_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#999999",
            markeredgecolor="#111111",
            markeredgewidth=1.4,
            color="#999999",
            label="included in decoding analysis",
            markersize=9,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#999999",
            color="#999999",
            alpha=1.0,
            label="excluded from decoding analysis",
            markersize=8,
        ),
    ]
    fig.legend(
        handles=group_handles + inclusion_handles,
        loc="upper center",
        ncol=min(4, len(group_handles) + len(inclusion_handles)),
        frameon=False,
    )
    fig.suptitle(
        "Session x Area Neuron Counts: all sessions shown, decoding inclusion highlighted",
        y=1.04,
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_neuron_counts(
    counts: pd.DataFrame,
    thresholds: pd.DataFrame,
    save_path: str | Path,
) -> None:
    if counts.empty:
        return

    areas = _ordered_areas(counts["area"])
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(areas),
        figsize=(5.2 * len(areas), 5.4),
        sharey=False,
        squeeze=False,
    )

    for ax, area in zip(axes.flat, areas):
        area_df = counts.loc[counts["area"] == area].copy()
        threshold_row = thresholds.loc[thresholds["area"] == area]
        threshold = threshold_row["balanced_neuron_threshold"].iloc[0] if not threshold_row.empty else np.nan

        for group in _ordered_groups(area_df["group"]):
            group_df = area_df.loc[area_df["group"] == group].sort_values("n_neurons", ascending=False).reset_index(drop=True)
            if group_df.empty:
                continue
            ax.plot(
                np.arange(1, len(group_df) + 1),
                group_df["n_neurons"],
                marker="o",
                linewidth=2,
                color=GROUP_PALETTE.get(group, "#444444"),
                label=group,
            )
        if np.isfinite(threshold):
            ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.5)

        ax.set_title(area)
        ax.set_xlabel("Session rank within group")
        ax.set_ylabel("Neurons per session")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Ranked Session Counts Per Area", y=1.04)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_decoding(
    frame: pd.DataFrame,
    task_order: list[str],
    save_path: str | Path,
    title: str,
) -> None:
    if frame.empty:
        return

    data = frame.copy()
    data["group"] = pd.Categorical(data["group"], categories=_ordered_groups(data["group"]), ordered=True)
    data["area"] = pd.Categorical(data["area"], categories=_ordered_areas(data["area"]), ordered=True)
    data["task"] = pd.Categorical(
        data["task"],
        categories=[task for task in task_order if task in data["task"].unique()],
        ordered=True,
    )

    tasks = [task for task in task_order if task in data["task"].unique()]
    areas = _ordered_areas(data["area"])
    if not tasks or not areas:
        return

    stimulus_offset_s = data["stimulus_offset_s"].dropna()
    stimulus_offset_s = float(stimulus_offset_s.iloc[0]) if not stimulus_offset_s.empty else 1.0

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=len(areas),
        ncols=len(tasks),
        figsize=(4.1 * len(tasks), 3.2 * len(areas)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, area in enumerate(areas):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            panel = data.loc[(data["area"] == area) & (data["task"] == task)]
            if panel.empty:
                ax.set_visible(False)
                continue

            sns.lineplot(
                data=panel,
                x="time_s",
                y="score",
                hue="group",
                estimator="mean",
                errorbar="se",
                linewidth=2,
                palette=GROUP_PALETTE,
                ax=ax,
            )

            chance_level = TASK_CHANCE_LEVELS.get(task)
            if chance_level is not None:
                ax.axhline(chance_level, color="#444444", linestyle="--", linewidth=1.1)
            ax.axvline(0.0, color="#111111", linestyle="-", linewidth=1.1)
            ax.axvline(stimulus_offset_s, color="#111111", linestyle=":", linewidth=1.1)

            if row_idx == 0:
                ax.set_title(task.replace("_", " "))
            else:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(f"{area}\nBalanced accuracy")
            else:
                ax.set_ylabel("")
            if row_idx == len(areas) - 1:
                ax.set_xlabel("Time from stimulus onset (s)")
            else:
                ax.set_xlabel("")
            ax.set_ylim(0, 1.02)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        if ax_handles:
            handles, labels = ax_handles, ax_labels
            break
    if handles:
        unique = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique.append((handle, label))
        fig.legend(
            [item[0] for item in unique],
            [item[1] for item in unique],
            loc="upper center",
            ncol=min(4, len(unique)),
            frameon=False,
        )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(input_dir)
    averaged = outputs["averaged"]
    position = outputs["position"]
    temporal = outputs["temporal"]
    counts = outputs["counts"]
    if counts.empty:
        counts = unique_neuron_counts(averaged, position)
    else:
        counts = counts.sort_values(["group", "area", "n_neurons"], ascending=[True, True, False])
    if args.skip_balanced:
        thresholds, inclusion = compute_decoding_inclusion(
            counts,
            min_neurons_per_session=args.balanced_min_neurons,
        )
    else:
        thresholds, inclusion = compute_balanced_thresholds(
            counts,
            min_sessions_per_group=args.balanced_min_sessions,
            min_neurons_per_session=args.balanced_min_neurons,
        )

    averaged = averaged.merge(
        inclusion[["group", "session_index", "session_name", "area", "balanced_included"]],
        on=["group", "session_index", "session_name", "area"],
        how="left",
    )
    position = position.merge(
        inclusion[["group", "session_index", "session_name", "area", "balanced_included"]],
        on=["group", "session_index", "session_name", "area"],
        how="left",
    )
    if not temporal.empty:
        temporal = temporal.merge(
            inclusion[["group", "session_index", "session_name", "area", "balanced_included"]],
            on=["group", "session_index", "session_name", "area"],
            how="left",
        )
    averaged["balanced_included"] = averaged["balanced_included"].fillna(False).astype(bool)
    position["balanced_included"] = position["balanced_included"].fillna(False).astype(bool)
    if not temporal.empty:
        temporal["balanced_included"] = temporal["balanced_included"].fillna(False).astype(bool)

    counts.to_csv(save_dir / "session_area_neuron_counts.csv", index=False)
    if not args.skip_balanced:
        inclusion.to_csv(save_dir / "session_area_balanced_inclusion.csv", index=False)
        balanced_averaged = outputs["balanced_averaged"]
        if balanced_averaged.empty:
            balanced_averaged = averaged.loc[averaged["balanced_included"]].copy()
        balanced_position = outputs["balanced_position"]
        if balanced_position.empty:
            balanced_position = position.loc[position["balanced_included"]].copy()
        balanced_temporal = outputs["balanced_temporal"]
        if balanced_temporal.empty and not temporal.empty:
            balanced_temporal = temporal.loc[temporal["balanced_included"]].copy()

        thresholds.to_csv(save_dir / "balanced_thresholds.csv", index=False)
        summarise_balanced_performance(balanced_averaged).to_csv(
            save_dir / "balanced_summary_population_averaged.csv",
            index=False,
        )
        summarise_balanced_performance(balanced_position).to_csv(
            save_dir / "balanced_summary_population_position.csv",
            index=False,
        )

    plot_performance(
        averaged,
        task_order=AVERAGED_TASK_ORDER,
        save_path=save_dir / "session_population_averaged_all.svg",
        title="Population Decoding From Averaged Trial Responses",
    )
    plot_performance(
        position,
        task_order=POSITION_TASK_ORDER,
        save_path=save_dir / "session_population_position_all.svg",
        title="Population Decoding Of Stimulus Position",
    )
    plot_neuron_count_diagnostic(
        inclusion,
        thresholds,
        save_dir / "session_area_neuron_count_diagnostic.svg",
    )
    plot_ranked_neuron_counts(
        inclusion,
        thresholds,
        save_dir / "session_area_neuron_count_ranked.svg",
    )
    plot_temporal_decoding(
        temporal,
        task_order=AVERAGED_TASK_ORDER,
        save_path=save_dir / "session_population_temporal.svg",
        title="Population Decoding Across Trial Time",
    )
    if not args.skip_balanced:
        plot_performance(
            balanced_averaged,
            task_order=AVERAGED_TASK_ORDER,
            save_path=save_dir / "session_population_averaged_balanced.svg",
            title="Population Decoding From Averaged Trial Responses (Balanced Sessions)",
        )
        plot_performance(
            balanced_position,
            task_order=POSITION_TASK_ORDER,
            save_path=save_dir / "session_population_position_balanced.svg",
            title="Population Decoding Of Stimulus Position (Balanced Sessions)",
        )
        plot_temporal_decoding(
            balanced_temporal,
            task_order=AVERAGED_TASK_ORDER,
            save_path=save_dir / "session_population_temporal_balanced.svg",
            title="Population Decoding Across Trial Time (Balanced Sessions)",
        )


if __name__ == "__main__":
    main()
