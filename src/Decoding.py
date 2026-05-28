from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


Reducer = str | Callable[..., np.ndarray]
TemporalMode = Literal["aggregate", "per-timepoint", "flatten"]
NeuronMode = Literal["population", "single-neuron"]
TaskType = Literal["classification", "regression"]


@dataclass
class DecodingResult:
    scores: np.ndarray
    mean_score: np.ndarray
    std_score: np.ndarray
    axes: tuple[str, ...]
    scoring: str | Callable[..., float] | None
    task: TaskType
    neuron_mode: NeuronMode
    temporal_mode: TemporalMode
    timepoints: np.ndarray | None = None
    neuron_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def scalar_score(self, reducer: Callable[[np.ndarray], float] = np.nanmean) -> float:
        values = np.asarray(self.mean_score, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(reducer(values))

    def to_frame(self) -> pd.DataFrame:
        if self.scores.ndim == 1:
            return pd.DataFrame(
                {
                    "mean_score": [float(self.mean_score)],
                    "std_score": [float(self.std_score)],
                }
            )

        leading_shape = self.scores.shape[:-1]
        rows = []
        for idx in np.ndindex(leading_shape):
            row = {
                axis_name: axis_value
                for axis_name, axis_value in zip(self.axes[:-1], idx)
            }
            fold_scores = self.scores[idx]
            row["mean_score"] = float(np.nanmean(fold_scores))
            row["std_score"] = float(np.nanstd(fold_scores))

            if "time" in row and self.timepoints is not None:
                row["timepoint"] = float(self.timepoints[row["time"]])
            if "neuron" in row and self.neuron_ids is not None:
                row["neuron_id"] = int(self.neuron_ids[row["neuron"]])
            rows.append(row)
        return pd.DataFrame(rows)


class Decoder:
    def __init__(
        self,
        estimator: BaseEstimator | type[BaseEstimator] | None = None,
        task: TaskType = "classification",
        scoring: str | Callable[..., float] | None = None,
        cv: int | Any = 5,
        random_state: int | None = 0,
        n_jobs: int | None = None,
    ):
        self.task = task
        self.estimator = estimator
        self.scoring = scoring or (
            "balanced_accuracy" if task == "classification" else "r2"
        )
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def stack_labeled_groups(
        grouped_signal: Mapping[Any, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        stacked_signal = []
        labels = []

        for label, signal in grouped_signal.items():
            signal = np.asarray(signal)
            if signal.ndim not in (2, 3):
                raise ValueError(
                    "Grouped signals should be 2D or 3D arrays with samples on axis 0."
                )
            stacked_signal.append(signal)
            labels.extend([label] * signal.shape[0])

        if not stacked_signal:
            raise ValueError("grouped_signal was empty.")

        return np.concatenate(stacked_signal, axis=0), np.asarray(labels, dtype=object)

    def decode(
        self,
        *,
        signal: np.ndarray | None = None,
        X: np.ndarray | None = None,
        labels: np.ndarray,
        groups: np.ndarray | None = None,
        neuron_mode: NeuronMode = "population",
        temporal_mode: TemporalMode = "aggregate",
        time_window: slice | tuple[int, int] | Iterable[int] | None = None,
        aggregation: Reducer = "mean",
        neuron_indices: np.ndarray | Iterable[int] | None = None,
        label_reducer: Literal["first", "last"] | Callable[[np.ndarray], Any] = "first",
    ) -> DecodingResult:
        labels = np.asarray(labels)

        if X is not None:
            scores = self._score_matrix(np.asarray(X), labels, groups=groups)
            return self._build_result(
                scores=scores,
                axes=("fold",),
                neuron_mode="population",
                temporal_mode="aggregate",
            )

        if signal is None:
            raise ValueError("Provide either `signal` or `X`.")

        signal = self._coerce_signal(np.asarray(signal), temporal_mode=temporal_mode)
        if signal.ndim == 2:
            labels_1d = self._reduce_labels(
                labels,
                time_indices=(
                    np.arange(labels.shape[1], dtype=int)
                    if labels.ndim == 2
                    else np.array([0], dtype=int)
                ),
                label_reducer=label_reducer,
            )
            neuron_ids = self._resolve_axis_indices(signal.shape[1], neuron_indices)
            return self._decode_aggregated(
                aggregated=signal,
                labels=labels_1d,
                groups=groups,
                neuron_ids=neuron_ids,
                neuron_mode=neuron_mode,
            )

        time_indices = self._resolve_axis_indices(signal.shape[1], time_window)
        neuron_ids = self._resolve_axis_indices(signal.shape[2], neuron_indices)

        if temporal_mode == "aggregate":
            aggregated = self._aggregate_signal(signal[:, time_indices, :], aggregation)
            labels_1d = self._reduce_labels(labels, time_indices, label_reducer)
            return self._decode_aggregated(
                aggregated=aggregated,
                labels=labels_1d,
                groups=groups,
                neuron_ids=neuron_ids,
                neuron_mode=neuron_mode,
            )

        if temporal_mode == "flatten":
            labels_1d = self._reduce_labels(labels, time_indices, label_reducer)
            flattened = signal[:, time_indices, :]
            return self._decode_flattened(
                flattened=flattened,
                labels=labels_1d,
                groups=groups,
                neuron_ids=neuron_ids,
                neuron_mode=neuron_mode,
            )

        return self._decode_temporal(
            signal=signal[:, time_indices, :],
            labels=labels,
            groups=groups,
            full_time_indices=time_indices,
            neuron_ids=neuron_ids,
            neuron_mode=neuron_mode,
        )

    def equal_neuron_subsampling(
        self,
        *,
        signal: np.ndarray,
        labels: np.ndarray,
        neuron_groups: Mapping[str, np.ndarray | Iterable[int]],
        n_boot: int = 100,
        n_neurons: int | None = None,
        summary_reducer: Callable[[np.ndarray], float] = np.nanmean,
        **decode_kwargs: Any,
    ) -> pd.DataFrame:
        group_indices = {
            name: np.asarray(indices, dtype=int)
            for name, indices in neuron_groups.items()
            if len(np.asarray(indices)) != 0
        }
        if not group_indices:
            return pd.DataFrame()

        if n_neurons is None:
            n_neurons = min(indices.size for indices in group_indices.values())

        rows = []
        for group_name, indices in group_indices.items():
            if indices.size < n_neurons:
                continue
            for boot in range(n_boot):
                sampled = np.sort(
                    self.rng.choice(indices, size=n_neurons, replace=False)
                )
                result = self.decode(
                    signal=signal,
                    labels=labels,
                    neuron_indices=sampled,
                    **decode_kwargs,
                )
                rows.append(
                    {
                        "group": group_name,
                        "bootstrap": boot,
                        "n_neurons": n_neurons,
                        "available_neurons": indices.size,
                        "score": result.scalar_score(summary_reducer),
                    }
                )
        return pd.DataFrame(rows)

    def _decode_aggregated(
        self,
        *,
        aggregated: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray | None,
        neuron_ids: np.ndarray,
        neuron_mode: NeuronMode,
    ) -> DecodingResult:
        aggregated = aggregated[:, neuron_ids]
        if neuron_mode == "population":
            scores = self._score_matrix(aggregated, labels, groups=groups)
            return self._build_result(
                scores=scores,
                axes=("fold",),
                neuron_mode=neuron_mode,
                temporal_mode="aggregate",
            )

        score_list = [
            self._score_matrix(aggregated[:, [i]], labels, groups=groups)
            for i in range(aggregated.shape[1])
        ]
        scores = self._pad_score_list(score_list)
        return self._build_result(
            scores=scores,
            axes=("neuron", "fold"),
            neuron_mode=neuron_mode,
            temporal_mode="aggregate",
            neuron_ids=neuron_ids,
        )

    def _decode_flattened(
        self,
        *,
        flattened: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray | None,
        neuron_ids: np.ndarray,
        neuron_mode: NeuronMode,
    ) -> DecodingResult:
        flattened = flattened[:, :, neuron_ids]
        if neuron_mode == "population":
            scores = self._score_matrix(
                flattened.reshape(flattened.shape[0], -1),
                labels,
                groups=groups,
            )
            return self._build_result(
                scores=scores,
                axes=("fold",),
                neuron_mode=neuron_mode,
                temporal_mode="flatten",
            )

        score_list = [
            self._score_matrix(flattened[:, :, i], labels, groups=groups)
            for i in range(flattened.shape[2])
        ]
        scores = self._pad_score_list(score_list)
        return self._build_result(
            scores=scores,
            axes=("neuron", "fold"),
            neuron_mode=neuron_mode,
            temporal_mode="flatten",
            neuron_ids=neuron_ids,
        )

    def _decode_temporal(
        self,
        *,
        signal: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray | None,
        full_time_indices: np.ndarray,
        neuron_ids: np.ndarray,
        neuron_mode: NeuronMode,
    ) -> DecodingResult:
        signal = signal[:, :, neuron_ids]
        if labels.ndim == 1:
            labels_by_time = np.repeat(labels[:, None], signal.shape[1], axis=1)
        else:
            labels_by_time = labels[:, full_time_indices]

        if neuron_mode == "population":
            score_list = [
                self._score_matrix(
                    signal[:, time_idx, :],
                    labels_by_time[:, time_idx],
                    groups=groups,
                )
                for time_idx in range(signal.shape[1])
            ]
            scores = self._pad_score_list(score_list)
            return self._build_result(
                scores=scores,
                axes=("time", "fold"),
                neuron_mode=neuron_mode,
                temporal_mode="per-timepoint",
                timepoints=full_time_indices,
            )

        temporal_scores = []
        for time_idx in range(signal.shape[1]):
            neuron_scores = [
                self._score_matrix(
                    signal[:, time_idx, neuron_idx][:, None],
                    labels_by_time[:, time_idx],
                    groups=groups,
                )
                for neuron_idx in range(signal.shape[2])
            ]
            temporal_scores.append(self._pad_score_list(neuron_scores))

        max_folds = max((scores.shape[-1] for scores in temporal_scores), default=1)
        stacked = np.full((signal.shape[1], signal.shape[2], max_folds), np.nan)
        for time_idx, scores in enumerate(temporal_scores):
            stacked[time_idx, :, : scores.shape[-1]] = scores

        return self._build_result(
            scores=stacked,
            axes=("time", "neuron", "fold"),
            neuron_mode=neuron_mode,
            temporal_mode="per-timepoint",
            timepoints=full_time_indices,
            neuron_ids=neuron_ids,
        )

    def _aggregate_signal(self, signal: np.ndarray, aggregation: Reducer) -> np.ndarray:
        if callable(aggregation):
            return aggregation(signal, axis=1)

        reducers = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "sum": np.nansum,
            "max": np.nanmax,
        }
        if aggregation not in reducers:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        return reducers[aggregation](signal, axis=1)

    def _reduce_labels(
        self,
        labels: np.ndarray,
        time_indices: np.ndarray,
        label_reducer: Literal["first", "last"] | Callable[[np.ndarray], Any],
    ) -> np.ndarray:
        if labels.ndim == 1:
            return labels

        labels = labels[:, time_indices]
        if callable(label_reducer):
            return np.apply_along_axis(label_reducer, 1, labels)
        if label_reducer == "first":
            return labels[:, 0]
        if label_reducer == "last":
            return labels[:, -1]
        raise ValueError(f"Unsupported label_reducer: {label_reducer}")

    def _score_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X[:, None]

        sample_mask = np.isfinite(X).all(axis=1)
        if y.dtype.kind in "fc":
            sample_mask &= np.isfinite(y)
        else:
            sample_mask &= pd.notna(y).to_numpy() if hasattr(pd.notna(y), "to_numpy") else pd.notna(y)

        X = X[sample_mask]
        y = y[sample_mask]
        if groups is not None:
            groups = np.asarray(groups)[sample_mask]

        if X.shape[0] < 2:
            return np.array([np.nan])
        if self.task == "classification" and np.unique(y).size < 2:
            return np.array([np.nan])

        cv = self._resolve_cv(y)
        if cv is None:
            return np.array([np.nan])

        try:
            return np.asarray(
                cross_val_score(
                    self._make_estimator(),
                    X,
                    y,
                    cv=cv,
                    groups=groups,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                ),
                dtype=float,
            )
        except Exception:
            return np.array([np.nan])

    def _resolve_cv(self, y: np.ndarray) -> Any | None:
        if self.cv is not None and not isinstance(self.cv, int):
            return self.cv

        desired_splits = 5 if self.cv is None else int(self.cv)
        if desired_splits < 2:
            return None

        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            n_splits = min(desired_splits, counts.min())
            if n_splits < 2:
                return None
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

        n_splits = min(desired_splits, y.shape[0])
        if n_splits < 2:
            return None
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def _make_estimator(self) -> BaseEstimator:
        estimator = self.estimator
        if estimator is None:
            estimator = self._default_estimator()

        if isinstance(estimator, type):
            estimator = estimator()
        return clone(estimator)

    def _default_estimator(self) -> BaseEstimator:
        if self.task == "classification":
            return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        return RidgeCV(alphas=np.logspace(-3, 3, 13))

    @staticmethod
    def _coerce_signal(signal: np.ndarray, temporal_mode: TemporalMode) -> np.ndarray:
        if signal.ndim == 3:
            return signal
        if signal.ndim == 2 and temporal_mode == "aggregate":
            return signal
        if signal.ndim == 2:
            return signal[:, :, None]
        raise ValueError("Signal should be a 2D or 3D numpy array.")

    @staticmethod
    def _resolve_axis_indices(
        axis_size: int,
        selection: slice | tuple[int, int] | Iterable[int] | np.ndarray | None,
    ) -> np.ndarray:
        if selection is None:
            return np.arange(axis_size, dtype=int)
        if isinstance(selection, slice):
            return np.arange(axis_size, dtype=int)[selection]
        if isinstance(selection, tuple) and len(selection) == 2:
            return np.arange(*selection, dtype=int)
        return np.asarray(list(selection), dtype=int)

    @staticmethod
    def _pad_score_list(score_list: list[np.ndarray]) -> np.ndarray:
        max_folds = max((scores.size for scores in score_list), default=1)
        padded = np.full((len(score_list), max_folds), np.nan)
        for row_idx, scores in enumerate(score_list):
            padded[row_idx, : scores.size] = scores
        return padded

    def _build_result(
        self,
        *,
        scores: np.ndarray,
        axes: tuple[str, ...],
        neuron_mode: NeuronMode,
        temporal_mode: TemporalMode,
        timepoints: np.ndarray | None = None,
        neuron_ids: np.ndarray | None = None,
    ) -> DecodingResult:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            mean_score = float(np.nanmean(scores))
            std_score = float(np.nanstd(scores))
        else:
            mean_score = np.nanmean(scores, axis=-1)
            std_score = np.nanstd(scores, axis=-1)

        return DecodingResult(
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            axes=axes,
            scoring=self.scoring,
            task=self.task,
            neuron_mode=neuron_mode,
            temporal_mode=temporal_mode,
            timepoints=None if timepoints is None else np.asarray(timepoints),
            neuron_ids=None if neuron_ids is None else np.asarray(neuron_ids),
        )
