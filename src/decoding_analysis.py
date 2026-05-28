from __future__ import annotations

import argparse
import __main__
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import PYDATA
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.Decoding import Decoder
from src.GLM_new import clean_group_signal
from src.VisualAreas import Areas


TRIAL_TYPES = OrderedDict(
    [
        ("Al", (0,)),
        ("AlVl", (1,)),
        ("AlVr", (2,)),
        ("Ar", (3,)),
        ("ArVl", (4,)),
        ("ArVr", (5,)),
        ("Vl", (6,)),
        ("Vr", (7,)),
    ]
)

AVERAGED_TASKS = OrderedDict(
    [
        ("visual_direction", OrderedDict([("left", (6, 1, 4)), ("right", (7, 2, 5))])),
        ("auditory_direction", OrderedDict([("left", (0, 1, 2)), ("right", (3, 4, 5))])),
        ("congruency", OrderedDict([("congruent", (1, 5)), ("incongruent", (2, 4))])),
        ("modality", OrderedDict([("V", (6, 7)), ("A", (0, 3)), ("AV", (1, 2, 4, 5))])),
        ("trial_type", TRIAL_TYPES),
    ]
)

POSITION_TASKS = {"V": (6, 7, 1, 2, 4, 5), "A": (0, 3, 1, 2, 4, 5)}
MIN_DECODING_NEURONS_PER_SESSION = 40


@dataclass
class GroupDataset:
    av: AUDVIS
    signal: np.ndarray
    areas: Areas


@dataclass
class SessionPopulation:
    group: str
    session_index: int
    session_name: str
    signal: np.ndarray
    trial_types: np.ndarray
    area_masks: dict[str, np.ndarray]


@dataclass
class BalancedSessionAreaPopulation:
    group: str
    session_index: int
    session_name: str
    signal: np.ndarray
    trial_types: np.ndarray
    area: str
    neuron_indices: np.ndarray
    available_neurons: int
    subsampled_neurons: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-post",
        default="pre",
        choices=("pre", "post", "both"),
    )
    parser.add_argument(
        "--signal-source",
        default="glm_clean",
        choices=("signal", "zsig", "glm_clean"),
    )
    parser.add_argument(
        "--areas",
        nargs="*",
        default=["V1", "AM/PM", "A/RL/AL", "LM"],
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--min-neurons", type=int, default=MIN_DECODING_NEURONS_PER_SESSION)
    parser.add_argument("--balanced-min-neurons", type=int, default=MIN_DECODING_NEURONS_PER_SESSION)
    parser.add_argument("--balanced-subsamples", type=int, default=1000)
    parser.add_argument("--balanced-n-jobs", type=int, default=1)
    parser.add_argument("--skip-balanced", action="store_true")
    parser.add_argument("--position-classes", type=int, default=15)
    parser.add_argument(
        "--save-dir",
        default=str(Path(PYDATA) / "decoding_population_sessionwise"),
    )
    return parser.parse_args()


def load_group_datasets(
    pre_post: Literal["pre", "post", "both"] = "pre",
    signal_source: Literal["signal", "zsig", "glm_clean"] = "signal",
) -> dict[str, GroupDataset]:
    __main__.Behavior = Behavior
    datasets = {}

    for av in load_in_data(pre_post=pre_post):
        datasets[av.NAME] = GroupDataset(
            av=av,
            signal=resolve_signal(av, signal_source=signal_source, pre_post=pre_post),
            areas=Areas(av, get_indices=True),
        )
    return datasets


def resolve_signal(
    av: AUDVIS,
    signal_source: Literal["signal", "zsig", "glm_clean"] = "signal",
    pre_post: Literal["pre", "post", "both"] = "pre",
) -> np.ndarray:
    if signal_source == "glm_clean":
        return clean_group_signal(group_name=av.NAME, pre_post=pre_post)
    if signal_source == "zsig":
        return av.baseline_correct_signal(av.zsig)
    return av.baseline_correct_signal(av.signal)


def iter_session_populations(
    dataset: GroupDataset,
    area_names: Iterable[str],
    min_neurons: int = 2,
) -> list[SessionPopulation]:
    populations = []
    requested_areas = list(area_names)

    for session_index, (start, stop) in enumerate(dataset.av.session_neurons):
        session_signal = dataset.signal[:, :, start:stop]
        session_trial_types = np.asarray(dataset.av.trials[session_index], dtype=int)
        session_name = dataset.av.sessions[session_index]["session"]
        session_global_neurons = np.arange(start, stop, dtype=int)
        area_masks = {}

        for area_name in requested_areas:
            if area_name == "all":
                local_indices = np.arange(stop - start, dtype=int)
            elif area_name in dataset.areas.area_indices:
                local_indices = np.intersect1d(
                    dataset.areas.area_indices[area_name],
                    session_global_neurons,
                ) - start
            else:
                continue

            if local_indices.size >= min_neurons:
                area_masks[area_name] = np.asarray(local_indices, dtype=int)

        if area_masks:
            populations.append(
                SessionPopulation(
                    group=dataset.av.NAME,
                    session_index=session_index,
                    session_name=session_name,
                    signal=session_signal,
                    trial_types=session_trial_types,
                    area_masks=area_masks,
                )
            )

    return populations


def session_area_neuron_counts(
    datasets: dict[str, GroupDataset],
    area_names: Iterable[str],
    signal_source: str,
    min_neurons: int = MIN_DECODING_NEURONS_PER_SESSION,
) -> pd.DataFrame:
    rows = []
    for population in [
        pop
        for dataset in datasets.values()
        for pop in iter_session_populations(dataset, area_names, min_neurons=1)
    ]:
        for area_name, neuron_indices in population.area_masks.items():
            n_neurons = int(neuron_indices.size)
            rows.append(
                {
                    "group": population.group,
                    "session_index": population.session_index,
                    "session_name": population.session_name,
                    "area": area_name,
                    "n_neurons": n_neurons,
                    "decoding_included": n_neurons >= min_neurons,
                    "min_decoding_neurons": int(min_neurons),
                    "signal_source": signal_source,
                }
            )
    return pd.DataFrame(rows)


def iter_balanced_session_area_populations(
    datasets: dict[str, GroupDataset],
    area_names: Iterable[str],
    min_neurons: int = 40,
) -> list[BalancedSessionAreaPopulation]:
    requested_groups = set(datasets)
    candidates = []

    for dataset in datasets.values():
        for population in iter_session_populations(dataset, area_names, min_neurons=1):
            for area_name, neuron_indices in population.area_masks.items():
                if neuron_indices.size < min_neurons:
                    continue
                candidates.append(
                    {
                        "population": population,
                        "area": area_name,
                        "neuron_indices": neuron_indices,
                        "n_neurons": int(neuron_indices.size),
                    }
                )

    balanced = []
    for area_name in sorted({row["area"] for row in candidates}):
        area_rows = [row for row in candidates if row["area"] == area_name]
        area_groups = {row["population"].group for row in area_rows}
        if not requested_groups.issubset(area_groups):
            continue

        subsampled_neurons = min(row["n_neurons"] for row in area_rows)
        for row in area_rows:
            population = row["population"]
            neuron_indices = np.asarray(row["neuron_indices"], dtype=int)
            balanced.append(
                BalancedSessionAreaPopulation(
                    group=population.group,
                    session_index=population.session_index,
                    session_name=population.session_name,
                    signal=population.signal,
                    trial_types=population.trial_types,
                    area=area_name,
                    neuron_indices=neuron_indices,
                    available_neurons=int(neuron_indices.size),
                    subsampled_neurons=int(subsampled_neurons),
                )
            )

    return balanced


def balanced_neuron_subsamples(
    population: BalancedSessionAreaPopulation,
    n_subsamples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if population.available_neurons <= population.subsampled_neurons:
        return [np.asarray(population.neuron_indices, dtype=int)]

    n_subsamples = max(1, int(n_subsamples))
    return [
        np.sort(
            rng.choice(
                population.neuron_indices,
                size=population.subsampled_neurons,
                replace=False,
            )
        )
        for _ in range(n_subsamples)
    ]


def stack_session_task_signal(
    session_signal: np.ndarray,
    session_trial_types: np.ndarray,
    label_groups: OrderedDict[str, tuple[int, ...]],
    neuron_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = OrderedDict()
    for label, trial_type_group in label_groups.items():
        trial_mask = np.isin(session_trial_types, trial_type_group)
        grouped[label] = session_signal[trial_mask][:, :, neuron_indices]
    return Decoder.stack_labeled_groups(grouped)


def position_labels_for_trial_type(
    trial_name: str,
    modality: Literal["V", "A"],
    n_timepoints: int,
    stimulus_window: tuple[int, int],
    n_positions: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    start, stop = map(int, stimulus_window)
    max_positions = min(n_positions, stop - start)
    active_frames = np.arange(start, start + max_positions, dtype=int)
    labels = np.full(n_timepoints, np.nan)

    position_sequence = np.arange(max_positions, dtype=int)
    direction_char = trial_name[trial_name.index(modality) + 1].lower()
    if direction_char == "r":
        position_sequence = position_sequence[::-1]

    labels[active_frames] = position_sequence
    return labels, active_frames


def build_session_position_dataset(
    population: SessionPopulation,
    modality: Literal["V", "A"],
    neuron_indices: np.ndarray,
    stimulus_window: tuple[int, int],
    n_positions: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    groups = []
    trial_counter = 0

    for tt in POSITION_TASKS[modality]:
        trial_mask = population.trial_types == tt
        trial_signal = population.signal[trial_mask][:, :, neuron_indices]

        if trial_signal.shape[0] == 0:
            continue

        trial_label_vector, active_frames = position_labels_for_trial_type(
            trial_name=population_trial_name(tt),
            modality=modality,
            n_timepoints=trial_signal.shape[1],
            stimulus_window=stimulus_window,
            n_positions=n_positions,
        )
        valid = np.isfinite(trial_label_vector)
        active_signal = trial_signal[:, valid, :]
        features.append(active_signal.reshape(-1, active_signal.shape[-1]))
        labels.append(np.tile(trial_label_vector[valid].astype(int), trial_signal.shape[0]))

        trial_ids = np.arange(trial_counter, trial_counter + trial_signal.shape[0], dtype=int)
        groups.append(np.repeat(trial_ids, valid.sum()))
        trial_counter += trial_signal.shape[0]

    if not features:
        return np.empty((0, neuron_indices.size)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    return np.vstack(features), np.concatenate(labels), np.concatenate(groups)


def population_trial_name(tt: int) -> str:
    return list(TRIAL_TYPES.keys())[tt]


def summarise_session_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    summary = (
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
    return summary


def trial_time_axis(av: AUDVIS, n_timepoints: int) -> np.ndarray:
    return (np.arange(n_timepoints, dtype=float) - float(av.TRIAL[0])) / float(av.SF)


def summarise_temporal_session_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    return (
        frame.groupby(
            [
                "group",
                "area",
                "task",
                "analysis",
                "signal_source",
                "scoring",
                "time_index",
                "time_s",
                "stimulus_offset_s",
            ],
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


def run_population_averaged_tasks(
    datasets: dict[str, GroupDataset],
    area_names: Iterable[str],
    signal_source: str,
    cv: int = 5,
    min_neurons: int = 2,
    random_state: int = 0,
) -> pd.DataFrame:
    decoder = Decoder(task="classification", cv=cv, random_state=random_state)
    rows = []

    for dataset in datasets.values():
        for population in iter_session_populations(dataset, area_names, min_neurons=min_neurons):
            for area_name, neuron_indices in population.area_masks.items():
                for task_name, label_groups in AVERAGED_TASKS.items():
                    signal, labels = stack_session_task_signal(
                        population.signal,
                        population.trial_types,
                        label_groups,
                        neuron_indices,
                    )
                    result = decoder.decode(
                        signal=signal,
                        labels=labels,
                        neuron_mode="population",
                        temporal_mode="aggregate",
                        time_window=tuple(dataset.av.TRIAL),
                    )
                    rows.append(
                        {
                            "group": population.group,
                            "session_index": population.session_index,
                            "session_name": population.session_name,
                            "area": area_name,
                            "task": task_name,
                            "analysis": "averaged_trial_response",
                            "score": result.scalar_score(),
                            "n_neurons": neuron_indices.size,
                            "n_samples": signal.shape[0],
                            "n_classes": np.unique(labels).size,
                            "signal_source": signal_source,
                            "scoring": str(decoder.scoring),
                        }
                    )

    return pd.DataFrame(rows)


def run_population_temporal_tasks(
    datasets: dict[str, GroupDataset],
    area_names: Iterable[str],
    signal_source: str,
    cv: int = 5,
    min_neurons: int = 2,
    random_state: int = 0,
) -> pd.DataFrame:
    decoder = Decoder(task="classification", cv=cv, random_state=random_state)
    rows = []

    for dataset in datasets.values():
        for population in iter_session_populations(dataset, area_names, min_neurons=min_neurons):
            time_s = trial_time_axis(dataset.av, population.signal.shape[1])
            stimulus_offset_s = (dataset.av.TRIAL[1] - dataset.av.TRIAL[0]) / dataset.av.SF

            for area_name, neuron_indices in population.area_masks.items():
                for task_name, label_groups in AVERAGED_TASKS.items():
                    signal, labels = stack_session_task_signal(
                        population.signal,
                        population.trial_types,
                        label_groups,
                        neuron_indices,
                    )
                    result = decoder.decode(
                        signal=signal,
                        labels=labels,
                        neuron_mode="population",
                        temporal_mode="per-timepoint",
                    )

                    scores = np.asarray(result.mean_score, dtype=float).ravel()
                    std_scores = np.asarray(result.std_score, dtype=float).ravel()
                    time_indices = (
                        np.asarray(result.timepoints, dtype=int)
                        if result.timepoints is not None
                        else np.arange(scores.size, dtype=int)
                    )

                    for row_idx, time_index in enumerate(time_indices):
                        rows.append(
                            {
                                "group": population.group,
                                "session_index": population.session_index,
                                "session_name": population.session_name,
                                "area": area_name,
                                "task": task_name,
                                "analysis": "time_resolved_trial_response",
                                "time_index": int(time_index),
                                "time_s": float(time_s[time_index]),
                                "stimulus_offset_s": float(stimulus_offset_s),
                                "score": float(scores[row_idx]),
                                "std_score": float(std_scores[row_idx]),
                                "n_neurons": neuron_indices.size,
                                "n_samples": signal.shape[0],
                                "n_classes": np.unique(labels).size,
                                "signal_source": signal_source,
                                "scoring": str(decoder.scoring),
                            }
                        )

    return pd.DataFrame(rows)


def _parallel_decode_balanced_jobs(
    jobs: list[tuple],
    worker,
    n_jobs: int,
) -> list:
    if not jobs:
        return []
    if int(n_jobs) == 1:
        return [worker(*job) for job in jobs]
    return Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
        delayed(worker)(*job) for job in jobs
    )


def _decode_balanced_averaged_job(
    population: BalancedSessionAreaPopulation,
    task_name: str,
    label_groups: OrderedDict[str, tuple[int, ...]],
    subsamples: list[np.ndarray],
    trial_window: tuple[int, int],
    signal_source: str,
    cv: int,
    random_state: int,
) -> dict:
    decoder = Decoder(task="classification", cv=cv, random_state=random_state)
    scores = []
    signal = labels = None
    for sampled_neurons in subsamples:
        signal, labels = stack_session_task_signal(
            population.signal,
            population.trial_types,
            label_groups,
            sampled_neurons,
        )
        result = decoder.decode(
            signal=signal,
            labels=labels,
            neuron_mode="population",
            temporal_mode="aggregate",
            time_window=trial_window,
        )
        scores.append(result.scalar_score())

    scores = np.asarray(scores, dtype=float)
    return {
        "group": population.group,
        "session_index": population.session_index,
        "session_name": population.session_name,
        "area": population.area,
        "task": task_name,
        "analysis": "balanced_averaged_trial_response",
        "score": float(np.nanmean(scores)),
        "std_score": float(np.nanstd(scores)),
        "n_neurons": population.subsampled_neurons,
        "available_neurons": population.available_neurons,
        "n_subsamples": len(subsamples),
        "n_samples": signal.shape[0],
        "n_classes": np.unique(labels).size,
        "signal_source": signal_source,
        "scoring": str(decoder.scoring),
    }


def _decode_balanced_temporal_job(
    population: BalancedSessionAreaPopulation,
    task_name: str,
    label_groups: OrderedDict[str, tuple[int, ...]],
    subsamples: list[np.ndarray],
    time_s: np.ndarray,
    stimulus_offset_s: float,
    signal_source: str,
    cv: int,
    random_state: int,
) -> list[dict]:
    decoder = Decoder(task="classification", cv=cv, random_state=random_state)
    score_list = []
    fold_std_list = []
    signal = labels = time_indices = None
    for sampled_neurons in subsamples:
        signal, labels = stack_session_task_signal(
            population.signal,
            population.trial_types,
            label_groups,
            sampled_neurons,
        )
        result = decoder.decode(
            signal=signal,
            labels=labels,
            neuron_mode="population",
            temporal_mode="per-timepoint",
        )
        score_list.append(np.asarray(result.mean_score, dtype=float).ravel())
        fold_std_list.append(np.asarray(result.std_score, dtype=float).ravel())
        time_indices = (
            np.asarray(result.timepoints, dtype=int)
            if result.timepoints is not None
            else np.arange(score_list[-1].size, dtype=int)
        )

    scores = np.vstack(score_list)
    fold_std_scores = np.vstack(fold_std_list)
    rows = []
    for row_idx, time_index in enumerate(time_indices):
        rows.append(
            {
                "group": population.group,
                "session_index": population.session_index,
                "session_name": population.session_name,
                "area": population.area,
                "task": task_name,
                "analysis": "balanced_time_resolved_trial_response",
                "time_index": int(time_index),
                "time_s": float(time_s[time_index]),
                "stimulus_offset_s": float(stimulus_offset_s),
                "score": float(np.nanmean(scores[:, row_idx])),
                "std_score": float(np.nanstd(scores[:, row_idx])),
                "fold_std_score": float(np.nanmean(fold_std_scores[:, row_idx])),
                "n_neurons": population.subsampled_neurons,
                "available_neurons": population.available_neurons,
                "n_subsamples": len(subsamples),
                "n_samples": signal.shape[0],
                "n_classes": np.unique(labels).size,
                "signal_source": signal_source,
                "scoring": str(decoder.scoring),
            }
        )
    return rows


def _decode_balanced_position_job(
    population: BalancedSessionAreaPopulation,
    modality: str,
    subsamples: list[np.ndarray],
    trial_window: tuple[int, int],
    n_positions: int,
    signal_source: str,
    cv: int,
    random_state: int,
) -> dict | None:
    scores = []
    X = labels = groups = None
    for sampled_neurons in subsamples:
        X, labels, groups = build_session_position_dataset(
            SessionPopulation(
                group=population.group,
                session_index=population.session_index,
                session_name=population.session_name,
                signal=population.signal,
                trial_types=population.trial_types,
                area_masks={population.area: sampled_neurons},
            ),
            modality=modality,
            neuron_indices=sampled_neurons,
            stimulus_window=trial_window,
            n_positions=n_positions,
        )
        if X.shape[0] == 0:
            continue

        n_splits = min(cv, np.unique(groups).size)
        if n_splits < 2:
            continue

        decoder = Decoder(
            task="classification",
            cv=GroupKFold(n_splits=n_splits),
            random_state=random_state,
        )
        result = decoder.decode(
            X=X,
            labels=labels,
            groups=groups,
        )
        scores.append(result.scalar_score())

    if not scores:
        return None

    scores = np.asarray(scores, dtype=float)
    return {
        "group": population.group,
        "session_index": population.session_index,
        "session_name": population.session_name,
        "area": population.area,
        "task": f"{modality}_position",
        "analysis": "balanced_timepoint_position",
        "score": float(np.nanmean(scores)),
        "std_score": float(np.nanstd(scores)),
        "n_neurons": population.subsampled_neurons,
        "available_neurons": population.available_neurons,
        "n_subsamples": len(scores),
        "n_samples": X.shape[0],
        "n_classes": np.unique(labels).size,
        "signal_source": signal_source,
        "scoring": str(decoder.scoring),
    }


def run_balanced_population_averaged_tasks(
    balanced_populations: list[BalancedSessionAreaPopulation],
    datasets: dict[str, GroupDataset],
    signal_source: str,
    cv: int = 5,
    n_subsamples: int = 1000,
    random_state: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    dataset_by_group = {dataset.av.NAME: dataset for dataset in datasets.values()}
    rng = np.random.default_rng(random_state)
    jobs = []
    for population in balanced_populations:
        dataset = dataset_by_group[population.group]
        subsamples = balanced_neuron_subsamples(population, n_subsamples, rng)
        trial_window = tuple(dataset.av.TRIAL)
        for task_name, label_groups in AVERAGED_TASKS.items():
            jobs.append(
                (
                    population,
                    task_name,
                    label_groups,
                    subsamples,
                    trial_window,
                    signal_source,
                    cv,
                    random_state,
                )
            )

    rows = _parallel_decode_balanced_jobs(
        jobs,
        _decode_balanced_averaged_job,
        n_jobs=n_jobs,
    )
    return pd.DataFrame(rows)


def run_balanced_population_temporal_tasks(
    balanced_populations: list[BalancedSessionAreaPopulation],
    datasets: dict[str, GroupDataset],
    signal_source: str,
    cv: int = 5,
    n_subsamples: int = 1000,
    random_state: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    dataset_by_group = {dataset.av.NAME: dataset for dataset in datasets.values()}
    rng = np.random.default_rng(random_state)
    jobs = []
    for population in balanced_populations:
        dataset = dataset_by_group[population.group]
        time_s = trial_time_axis(dataset.av, population.signal.shape[1])
        stimulus_offset_s = (dataset.av.TRIAL[1] - dataset.av.TRIAL[0]) / dataset.av.SF
        subsamples = balanced_neuron_subsamples(population, n_subsamples, rng)
        for task_name, label_groups in AVERAGED_TASKS.items():
            jobs.append(
                (
                    population,
                    task_name,
                    label_groups,
                    subsamples,
                    time_s,
                    stimulus_offset_s,
                    signal_source,
                    cv,
                    random_state,
                )
            )

    row_groups = _parallel_decode_balanced_jobs(
        jobs,
        _decode_balanced_temporal_job,
        n_jobs=n_jobs,
    )
    rows = [row for group in row_groups for row in group]
    return pd.DataFrame(rows)


def run_balanced_population_position_tasks(
    balanced_populations: list[BalancedSessionAreaPopulation],
    datasets: dict[str, GroupDataset],
    signal_source: str,
    cv: int = 5,
    n_positions: int = 15,
    n_subsamples: int = 1000,
    random_state: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    dataset_by_group = {dataset.av.NAME: dataset for dataset in datasets.values()}
    rng = np.random.default_rng(random_state)
    jobs = []

    for population in balanced_populations:
        dataset = dataset_by_group[population.group]
        subsamples = balanced_neuron_subsamples(population, n_subsamples, rng)
        for modality in ("V", "A"):
            jobs.append(
                (
                    population,
                    modality,
                    subsamples,
                    tuple(dataset.av.TRIAL),
                    n_positions,
                    signal_source,
                    cv,
                    random_state,
                )
            )

    rows = [
        row
        for row in _parallel_decode_balanced_jobs(
            jobs,
            _decode_balanced_position_job,
            n_jobs=n_jobs,
        )
        if row is not None
    ]
    return pd.DataFrame(rows)


def run_population_position_tasks(
    datasets: dict[str, GroupDataset],
    area_names: Iterable[str],
    signal_source: str,
    cv: int = 5,
    min_neurons: int = 2,
    n_positions: int = 15,
    random_state: int = 0,
) -> pd.DataFrame:
    rows = []

    for dataset in datasets.values():
        for population in iter_session_populations(dataset, area_names, min_neurons=min_neurons):
            for area_name, neuron_indices in population.area_masks.items():
                for modality in ("V", "A"):
                    X, labels, groups = build_session_position_dataset(
                        population,
                        modality=modality,
                        neuron_indices=neuron_indices,
                        stimulus_window=tuple(dataset.av.TRIAL),
                        n_positions=n_positions,
                    )
                    if X.shape[0] == 0:
                        continue

                    n_splits = min(cv, np.unique(groups).size)
                    if n_splits < 2:
                        continue

                    decoder = Decoder(
                        task="classification",
                        cv=GroupKFold(n_splits=n_splits),
                        random_state=random_state,
                    )
                    result = decoder.decode(
                        X=X,
                        labels=labels,
                        groups=groups,
                    )
                    rows.append(
                        {
                            "group": population.group,
                            "session_index": population.session_index,
                            "session_name": population.session_name,
                            "area": area_name,
                            "task": f"{modality}_position",
                            "analysis": "timepoint_position",
                            "score": result.scalar_score(),
                            "n_neurons": neuron_indices.size,
                            "n_samples": X.shape[0],
                            "n_classes": np.unique(labels).size,
                            "signal_source": signal_source,
                            "scoring": str(decoder.scoring),
                        }
                    )

    return pd.DataFrame(rows)


def save_results(results: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        if not frame.empty:
            frame.to_csv(save_dir / f"{name}.csv", index=False)


def run_all_population_analyses(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    datasets = load_group_datasets(
        pre_post=args.pre_post,
        signal_source=args.signal_source,
    )
    counts = session_area_neuron_counts(
        datasets=datasets,
        area_names=args.areas,
        signal_source=args.signal_source,
        min_neurons=args.min_neurons,
    )

    print("Running full-trial averaged population decoding", flush=True)
    averaged_session = run_population_averaged_tasks(
        datasets=datasets,
        area_names=args.areas,
        signal_source=args.signal_source,
        cv=args.cv,
        min_neurons=args.min_neurons,
    )
    print("Running time-resolved population decoding", flush=True)
    temporal_session = run_population_temporal_tasks(
        datasets=datasets,
        area_names=args.areas,
        signal_source=args.signal_source,
        cv=args.cv,
        min_neurons=args.min_neurons,
    )
    print("Running position population decoding", flush=True)
    position_session = run_population_position_tasks(
        datasets=datasets,
        area_names=args.areas,
        signal_source=args.signal_source,
        cv=args.cv,
        min_neurons=args.min_neurons,
        n_positions=args.position_classes,
    )

    results = {
        "session_population_averaged": averaged_session,
        "summary_population_averaged": summarise_session_results(averaged_session),
        "session_population_temporal": temporal_session,
        "summary_population_temporal": summarise_temporal_session_results(temporal_session),
        "session_population_position": position_session,
        "summary_population_position": summarise_session_results(position_session),
        "session_area_neuron_counts": counts,
    }

    if not args.skip_balanced:
        balanced_populations = iter_balanced_session_area_populations(
            datasets=datasets,
            area_names=args.areas,
            min_neurons=args.balanced_min_neurons,
        )
        print("Running balanced averaged population decoding", flush=True)
        balanced_averaged_session = run_balanced_population_averaged_tasks(
            balanced_populations=balanced_populations,
            datasets=datasets,
            signal_source=args.signal_source,
            cv=args.cv,
            n_subsamples=args.balanced_subsamples,
            n_jobs=args.balanced_n_jobs,
        )
        print("Running balanced time-resolved population decoding", flush=True)
        balanced_temporal_session = run_balanced_population_temporal_tasks(
            balanced_populations=balanced_populations,
            datasets=datasets,
            signal_source=args.signal_source,
            cv=args.cv,
            n_subsamples=args.balanced_subsamples,
            n_jobs=args.balanced_n_jobs,
        )
        print("Running balanced position population decoding", flush=True)
        balanced_position_session = run_balanced_population_position_tasks(
            balanced_populations=balanced_populations,
            datasets=datasets,
            signal_source=args.signal_source,
            cv=args.cv,
            n_positions=args.position_classes,
            n_subsamples=args.balanced_subsamples,
            n_jobs=args.balanced_n_jobs,
        )
        results.update(
            {
                "balanced_session_population_averaged": balanced_averaged_session,
                "balanced_summary_population_averaged": summarise_session_results(balanced_averaged_session),
                "balanced_session_population_temporal": balanced_temporal_session,
                "balanced_summary_population_temporal": summarise_temporal_session_results(balanced_temporal_session),
                "balanced_session_population_position": balanced_position_session,
                "balanced_summary_population_position": summarise_session_results(balanced_position_session),
            }
        )
    print(f"Saving decoding outputs to {args.save_dir}", flush=True)
    save_results(results, args.save_dir)
    return results


def main() -> dict[str, pd.DataFrame]:
    args = parse_args()
    return run_all_population_analyses(args)


if __name__ == "__main__":
    main()
