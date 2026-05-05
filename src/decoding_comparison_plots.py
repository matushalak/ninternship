from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import PLOTSDIR, PYDATA
from src.decoding_plots import (
    AREA_ORDER,
    AVERAGED_TASK_ORDER,
    POSITION_TASK_ORDER,
    TASK_CHANCE_LEVELS,
)


CONDITION_ORDER = ["pre", "post"]
GROUP_BASE_ORDER = ["g1", "g2"]
GROUP_BASE_PALETTE = {
    "g1": "#2C6E91",
    "g2": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pydata-root",
        default=str(Path(PYDATA) / "decoding_population_sessionwise"),
    )
    parser.add_argument(
        "--plots-root",
        default=str(Path(PLOTSDIR) / "decoding_population_sessionwise"),
    )
    parser.add_argument(
        "--save-pydata-dir",
        default=str(Path(PYDATA) / "decoding_population_sessionwise" / "comparison_glm_clean"),
    )
    parser.add_argument(
        "--save-plots-dir",
        default=str(Path(PLOTSDIR) / "decoding_population_sessionwise" / "comparison_glm_clean"),
    )
    return parser.parse_args()


def _add_condition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    parsed = data["group"].str.extract(r"^(g\d)(pre|post)$")
    data["group_base"] = parsed[0]
    data["condition"] = parsed[1]
    data["group_base"] = pd.Categorical(data["group_base"], categories=GROUP_BASE_ORDER, ordered=True)
    data["condition"] = pd.Categorical(data["condition"], categories=CONDITION_ORDER, ordered=True)
    data["area"] = pd.Categorical(data["area"], categories=[a for a in AREA_ORDER if a in data["area"].unique()], ordered=True)
    return data


def load_condition_outputs(pydata_root: str | Path, plots_root: str | Path) -> dict[str, pd.DataFrame]:
    pydata_root = Path(pydata_root)
    plots_root = Path(plots_root)
    outputs: dict[str, list[pd.DataFrame]] = {
        "averaged_session": [],
        "averaged_summary": [],
        "position_session": [],
        "position_summary": [],
        "counts": [],
    }

    for condition in CONDITION_ORDER:
        data_dir = pydata_root / f"{condition}_glm_clean"
        plot_dir = plots_root / f"{condition}_glm_clean"

        averaged_session = pd.read_csv(data_dir / "session_population_averaged.csv")
        averaged_summary = pd.read_csv(data_dir / "summary_population_averaged.csv")
        position_session = pd.read_csv(data_dir / "session_population_position.csv")
        position_summary = pd.read_csv(data_dir / "summary_population_position.csv")
        counts = pd.read_csv(plot_dir / "session_area_neuron_counts.csv")

        outputs["averaged_session"].append(averaged_session)
        outputs["averaged_summary"].append(averaged_summary)
        outputs["position_session"].append(position_session)
        outputs["position_summary"].append(position_summary)
        outputs["counts"].append(counts)

    return {name: _add_condition_columns(pd.concat(frames, ignore_index=True)) for name, frames in outputs.items()}


def summarise_counts(counts: pd.DataFrame) -> pd.DataFrame:
    return (
        counts.groupby(
            ["group", "group_base", "condition", "area", "signal_source"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_neurons=("n_neurons", "mean"),
            std_neurons=("n_neurons", "std"),
            sem_neurons=("n_neurons", "sem"),
            min_neurons=("n_neurons", "min"),
            max_neurons=("n_neurons", "max"),
            n_sessions=("session_name", "nunique"),
        )
    )


def compute_post_minus_pre(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = (
        summary.pivot_table(
            index=["group_base", "area", "task"],
            columns="condition",
            values=value_col,
            observed=True,
        )
        .reset_index()
    )
    wide["delta_post_minus_pre"] = wide["post"] - wide["pre"]
    return wide


def compute_g2_minus_g1(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = (
        summary.pivot_table(
            index=["condition", "area", "task"],
            columns="group_base",
            values=value_col,
            observed=True,
        )
        .reset_index()
    )
    wide["delta_g2_minus_g1"] = wide["g2"] - wide["g1"]
    return wide


def compute_count_post_minus_pre(summary: pd.DataFrame) -> pd.DataFrame:
    wide = (
        summary.pivot_table(
            index=["group_base", "area"],
            columns="condition",
            values="mean_neurons",
            observed=True,
        )
        .reset_index()
    )
    wide["delta_post_minus_pre"] = wide["post"] - wide["pre"]
    return wide


def compute_count_g2_minus_g1(summary: pd.DataFrame) -> pd.DataFrame:
    wide = (
        summary.pivot_table(
            index=["condition", "area"],
            columns="group_base",
            values="mean_neurons",
            observed=True,
        )
        .reset_index()
    )
    wide["delta_g2_minus_g1"] = wide["g2"] - wide["g1"]
    return wide


def plot_session_grid(
    frame: pd.DataFrame,
    task_order: list[str],
    save_path: str | Path,
    title: str,
    y_col: str = "score",
    y_label: str = "Balanced accuracy",
    chance_levels: dict[str, float] | None = None,
) -> None:
    if frame.empty:
        return

    tasks = [task for task in task_order if task in frame["task"].unique()]
    areas = [area for area in AREA_ORDER if area in frame["area"].unique()]
    if not tasks or not areas:
        return

    data = frame.copy()
    data["task"] = pd.Categorical(data["task"], categories=tasks, ordered=True)
    data["area"] = pd.Categorical(data["area"], categories=areas, ordered=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=len(areas),
        ncols=len(tasks),
        figsize=(4.0 * len(tasks), 3.8 * len(areas)),
        sharey=False,
        squeeze=False,
    )

    for row_idx, area in enumerate(areas):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            panel = data.loc[(data["area"] == area) & (data["task"] == task)].copy()
            if panel.empty:
                ax.set_visible(False)
                continue

            sns.stripplot(
                data=panel,
                x="condition",
                y=y_col,
                hue="group_base",
                dodge=True,
                jitter=0.16,
                alpha=0.55,
                size=5.5,
                palette=GROUP_BASE_PALETTE,
                ax=ax,
            )
            sns.pointplot(
                data=panel,
                x="condition",
                y=y_col,
                hue="group_base",
                dodge=0.35,
                errorbar="se",
                markers="D",
                linestyles="-",
                palette=GROUP_BASE_PALETTE,
                ax=ax,
            )

            if chance_levels is not None:
                chance = chance_levels.get(task)
                if chance is not None:
                    ax.axhline(chance, color="#444444", linestyle="--", linewidth=1.2, zorder=1)
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(min(ymin, chance - 0.03), ymax)

            if row_idx == 0:
                ax.set_title(task.replace("_", " "))
            else:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(f"{area}\n{y_label}")
            else:
                ax.set_ylabel("")
            if row_idx == len(areas) - 1:
                ax.set_xlabel("Condition")
            else:
                ax.set_xlabel("")

            if ax.get_legend() is not None:
                ax.get_legend().remove()

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
            ncol=len(unique),
            frameon=False,
        )

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_neuron_count_grid(counts: pd.DataFrame, save_path: str | Path, title: str) -> None:
    if counts.empty:
        return

    areas = [area for area in AREA_ORDER if area in counts["area"].unique()]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(areas),
        figsize=(4.4 * len(areas), 4.8),
        sharey=False,
        squeeze=False,
    )

    for ax, area in zip(axes.flat, areas):
        panel = counts.loc[counts["area"] == area].copy()
        sns.stripplot(
            data=panel,
            x="condition",
            y="n_neurons",
            hue="group_base",
            dodge=True,
            jitter=0.16,
            alpha=0.6,
            size=5.5,
            palette=GROUP_BASE_PALETTE,
            ax=ax,
        )
        sns.pointplot(
            data=panel,
            x="condition",
            y="n_neurons",
            hue="group_base",
            dodge=0.35,
            errorbar="se",
            markers="D",
            linestyles="-",
            palette=GROUP_BASE_PALETTE,
            ax=ax,
        )
        ax.set_title(area)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Neurons per session")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

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
            ncol=len(unique),
            frameon=False,
        )

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_difference_heatmap(
    frame: pd.DataFrame,
    row_key: str,
    value_col: str,
    panel_order: list[str],
    task_order: list[str],
    save_path: str | Path,
    title: str,
    cmap: str = "coolwarm",
) -> None:
    if frame.empty:
        return

    panels = [panel for panel in panel_order if panel in frame[row_key].unique()]
    tasks = [task for task in task_order if task in frame["task"].unique()]
    areas = [area for area in AREA_ORDER if area in frame["area"].unique()]
    if not panels or not tasks or not areas:
        return

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        figsize=(4.0 * len(panels), 5.0),
        squeeze=False,
    )

    vmax = np.nanmax(np.abs(frame[value_col].to_numpy()))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    for ax, panel in zip(axes.flat, panels):
        panel_df = frame.loc[frame[row_key] == panel].copy()
        heat = (
            panel_df.pivot(index="area", columns="task", values=value_col)
            .reindex(index=areas, columns=tasks)
        )
        sns.heatmap(
            heat,
            ax=ax,
            cmap=cmap,
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar=ax is axes.flat[-1],
        )
        ax.set_title(str(panel))
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_count_difference_bars(
    frame: pd.DataFrame,
    panel_key: str,
    value_col: str,
    panel_order: list[str],
    save_path: str | Path,
    title: str,
) -> None:
    if frame.empty:
        return

    panels = [panel for panel in panel_order if panel in frame[panel_key].unique()]
    areas = [area for area in AREA_ORDER if area in frame["area"].unique()]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        figsize=(4.3 * len(panels), 4.8),
        squeeze=False,
        sharey=True,
    )

    for ax, panel in zip(axes.flat, panels):
        panel_df = frame.loc[frame[panel_key] == panel].copy()
        panel_df["area"] = pd.Categorical(panel_df["area"], categories=areas, ordered=True)
        sns.barplot(
            data=panel_df,
            x="area",
            y=value_col,
            hue="area",
            palette="coolwarm",
            dodge=False,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.axhline(0.0, color="#333333", linewidth=1.2)
        ax.set_title(str(panel))
        ax.set_xlabel("")
        ax.set_ylabel("Mean neuron count delta")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    save_pydata_dir = Path(args.save_pydata_dir)
    save_plots_dir = Path(args.save_plots_dir)
    save_pydata_dir.mkdir(parents=True, exist_ok=True)
    save_plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_condition_outputs(args.pydata_root, args.plots_root)
    averaged_session = data["averaged_session"]
    averaged_summary = data["averaged_summary"]
    position_session = data["position_session"]
    position_summary = data["position_summary"]
    counts = data["counts"]
    count_summary = summarise_counts(counts)

    averaged_summary.to_csv(save_pydata_dir / "comparison_summary_population_averaged.csv", index=False)
    position_summary.to_csv(save_pydata_dir / "comparison_summary_population_position.csv", index=False)
    count_summary.to_csv(save_pydata_dir / "comparison_summary_neuron_counts.csv", index=False)

    avg_post_pre = compute_post_minus_pre(averaged_summary, "mean_score")
    pos_post_pre = compute_post_minus_pre(position_summary, "mean_score")
    avg_g2_g1 = compute_g2_minus_g1(averaged_summary, "mean_score")
    pos_g2_g1 = compute_g2_minus_g1(position_summary, "mean_score")
    count_post_pre = compute_count_post_minus_pre(count_summary)
    count_g2_g1 = compute_count_g2_minus_g1(count_summary)

    avg_post_pre.to_csv(save_pydata_dir / "delta_post_minus_pre_population_averaged.csv", index=False)
    pos_post_pre.to_csv(save_pydata_dir / "delta_post_minus_pre_population_position.csv", index=False)
    avg_g2_g1.to_csv(save_pydata_dir / "delta_g2_minus_g1_population_averaged.csv", index=False)
    pos_g2_g1.to_csv(save_pydata_dir / "delta_g2_minus_g1_population_position.csv", index=False)
    count_post_pre.to_csv(save_pydata_dir / "delta_post_minus_pre_neuron_counts.csv", index=False)
    count_g2_g1.to_csv(save_pydata_dir / "delta_g2_minus_g1_neuron_counts.csv", index=False)

    plot_session_grid(
        averaged_session,
        task_order=AVERAGED_TASK_ORDER,
        save_path=save_plots_dir / "comparison_population_averaged_session_grid.svg",
        title="Population Decoding: Pre vs Post Across Groups",
        chance_levels=TASK_CHANCE_LEVELS,
    )
    plot_session_grid(
        position_session,
        task_order=POSITION_TASK_ORDER,
        save_path=save_plots_dir / "comparison_population_position_session_grid.svg",
        title="Position Decoding: Pre vs Post Across Groups",
        chance_levels=TASK_CHANCE_LEVELS,
    )
    plot_neuron_count_grid(
        counts,
        save_path=save_plots_dir / "comparison_neuron_counts_session_grid.svg",
        title="Neuron Counts Per Session-Area: Pre vs Post Across Groups",
    )

    plot_difference_heatmap(
        avg_post_pre,
        row_key="group_base",
        value_col="delta_post_minus_pre",
        panel_order=GROUP_BASE_ORDER,
        task_order=AVERAGED_TASK_ORDER,
        save_path=save_plots_dir / "difference_heatmap_post_minus_pre_population_averaged.svg",
        title="Decoding Change (Post - Pre) by Group",
    )
    plot_difference_heatmap(
        pos_post_pre,
        row_key="group_base",
        value_col="delta_post_minus_pre",
        panel_order=GROUP_BASE_ORDER,
        task_order=POSITION_TASK_ORDER,
        save_path=save_plots_dir / "difference_heatmap_post_minus_pre_population_position.svg",
        title="Position Decoding Change (Post - Pre) by Group",
    )
    plot_difference_heatmap(
        avg_g2_g1,
        row_key="condition",
        value_col="delta_g2_minus_g1",
        panel_order=CONDITION_ORDER,
        task_order=AVERAGED_TASK_ORDER,
        save_path=save_plots_dir / "difference_heatmap_g2_minus_g1_population_averaged.svg",
        title="Between-Group Difference (g2 - g1) by Condition",
    )
    plot_difference_heatmap(
        pos_g2_g1,
        row_key="condition",
        value_col="delta_g2_minus_g1",
        panel_order=CONDITION_ORDER,
        task_order=POSITION_TASK_ORDER,
        save_path=save_plots_dir / "difference_heatmap_g2_minus_g1_population_position.svg",
        title="Position Between-Group Difference (g2 - g1) by Condition",
    )
    plot_count_difference_bars(
        count_post_pre,
        panel_key="group_base",
        value_col="delta_post_minus_pre",
        panel_order=GROUP_BASE_ORDER,
        save_path=save_plots_dir / "difference_bars_post_minus_pre_neuron_counts.svg",
        title="Neuron Count Change (Post - Pre) by Group",
    )
    plot_count_difference_bars(
        count_g2_g1,
        panel_key="condition",
        value_col="delta_g2_minus_g1",
        panel_order=CONDITION_ORDER,
        save_path=save_plots_dir / "difference_bars_g2_minus_g1_neuron_counts.svg",
        title="Between-Group Neuron Count Difference (g2 - g1)",
    )


if __name__ == "__main__":
    main()
