from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, Mapping

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from src.AUDVIS import AUDVIS


DEFAULT_TRIAL_GROUPS = OrderedDict(
    [
        ("V", (6, 7)),
        ("A", (0, 3)),
        ("AV+", (1, 5)),
        ("AV-", (2, 4)),
    ]
)

PopulationMode = Literal["neuronal_embeddings", "neural_manifold"]


@dataclass(frozen=True)
class SessionSlice:
    group: str
    session_index: int
    session_name: str
    signal: np.ndarray
    trial_types: np.ndarray
    neuron_ids: np.ndarray


@dataclass
class SessionPCAResult:
    mode: PopulationMode
    group: str
    session_index: int
    session_name: str
    matrix: np.ndarray
    scores: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    sample_metadata: dict[str, np.ndarray]
    feature_metadata: dict[str, np.ndarray]
    pca: PCA


def normalize_trial_groups(
    trial_groups: Mapping[str, Iterable[int]] | None,
    trial_types: np.ndarray,
    trial_label_map: Mapping[int, str] | None = None,
) -> OrderedDict[str, tuple[int, ...]]:
    if trial_groups is None:
        return OrderedDict(
            (
                str(trial_label_map[int(tt)]) if trial_label_map and int(tt) in trial_label_map else str(int(tt)),
                (int(tt),),
            )
            for tt in np.unique(np.asarray(trial_types, dtype=int))
        )

    return OrderedDict(
        (str(label), tuple(int(tt) for tt in tts))
        for label, tts in trial_groups.items()
    )


def _resolve_n_components(
    requested: int | float | None,
    matrix_shape: tuple[int, int],
) -> int | float | None:
    if requested is None or isinstance(requested, float):
        return requested

    max_components = min(matrix_shape)
    if max_components < 1:
        raise ValueError("Cannot fit PCA on an empty matrix.")
    return min(int(requested), max_components)


class PopulationAnalysis:
    """
    Session-wise population analyses for an AUDVIS dataset.

    The two PCA variants intentionally stay separate:
    - neuronal embeddings: PCA on rows = neurons
    - neural manifold: PCA on rows = population states

    Since neuron identities and counts differ between sessions, PCA is fit
    independently in each session and results are returned per session.
    """

    def __init__(
        self,
        av: AUDVIS,
        signal: np.ndarray | None = None,
        baseline_correct: bool = True,
    ):
        self.av = av
        raw_signal = av.zsig if signal is None else np.asarray(signal)
        self.signal = av.baseline_correct_signal(raw_signal) if baseline_correct else raw_signal
        self.trial_label_map = {
            int(k): str(v)
            for k, v in getattr(av, "int_to_str_trials_map", {}).items()
        }
        self.fit_time_window = tuple(int(x) for x in getattr(av, "TRIAL", (0, raw_signal.shape[1])))

    def iter_sessions(
        self,
        session_indices: Iterable[int] | None = None,
    ) -> list[SessionSlice]:
        if session_indices is None:
            session_indices = range(len(self.av.session_neurons))

        sessions = []
        for session_index in session_indices:
            start, stop = self.av.session_neurons[session_index]
            sessions.append(
                SessionSlice(
                    group=self.av.NAME,
                    session_index=session_index,
                    session_name=self.av.sessions[session_index]["session"],
                    signal=self.signal[:, :, start:stop],
                    trial_types=np.asarray(self.av.trials[session_index], dtype=int),
                    neuron_ids=np.arange(start, stop, dtype=int),
                )
            )
        return sessions

    @staticmethod
    def _select_neurons(
        session: SessionSlice,
        neuron_indices: np.ndarray | Iterable[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if neuron_indices is None:
            neuron_indices = np.arange(session.signal.shape[-1], dtype=int)
        else:
            neuron_indices = np.asarray(list(neuron_indices), dtype=int)

        return session.signal[:, :, neuron_indices], session.neuron_ids[neuron_indices]

    @staticmethod
    def _group_session_signal(
        session_signal: np.ndarray,
        trial_types: np.ndarray,
        trial_groups: OrderedDict[str, tuple[int, ...]],
    ) -> OrderedDict[str, np.ndarray]:
        grouped = OrderedDict()
        for label, tt_group in trial_groups.items():
            grouped[label] = session_signal[np.isin(trial_types, tt_group)]
        return grouped

    @staticmethod
    def _resolve_time_indices(
        n_timepoints: int,
        time_window: tuple[int, int] | slice | Iterable[int] | None,
    ) -> np.ndarray:
        if time_window is None:
            return np.arange(n_timepoints, dtype=int)
        if isinstance(time_window, slice):
            return np.arange(n_timepoints, dtype=int)[time_window]
        if isinstance(time_window, tuple):
            start, stop = map(int, time_window)
            return np.arange(start, stop, dtype=int)
        return np.asarray(list(time_window), dtype=int)

    @staticmethod
    def _build_embedding_matrix(
        grouped_signal: Mapping[str, np.ndarray],
        neuron_ids: np.ndarray,
        time_indices: np.ndarray,
        average_trials: bool = True,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        blocks = []
        feature_groups = []
        feature_trials = []
        feature_time = []

        for label, signal in grouped_signal.items():
            if signal.shape[0] == 0:
                continue

            if average_trials:
                mean_signal = np.nanmean(signal[:, time_indices, :], axis=0)
                block = mean_signal.T
                blocks.append(block)
                feature_groups.extend([label] * block.shape[1])
                feature_trials.extend([-1] * block.shape[1])
                feature_time.extend(time_indices)
                continue

            selected_signal = signal[:, time_indices, :]
            block = selected_signal.transpose(2, 0, 1).reshape(selected_signal.shape[-1], -1)
            blocks.append(block)
            for trial_index in range(selected_signal.shape[0]):
                feature_groups.extend([label] * selected_signal.shape[1])
                feature_trials.extend([trial_index] * selected_signal.shape[1])
                feature_time.extend(time_indices)

        if not blocks:
            raise ValueError("No signal remained after trial grouping.")

        matrix = np.concatenate(blocks, axis=1)
        valid_features = np.isfinite(matrix).all(axis=0)
        matrix = matrix[:, valid_features]
        valid_samples = np.isfinite(matrix).all(axis=1)
        matrix = matrix[valid_samples]

        sample_metadata = {
            "neuron_id": neuron_ids[valid_samples],
        }
        feature_metadata = {
            "group": np.asarray(feature_groups, dtype=object)[valid_features],
            "trial_index": np.asarray(feature_trials, dtype=int)[valid_features],
            "time_index": np.asarray(feature_time, dtype=int)[valid_features],
        }
        return matrix, sample_metadata, feature_metadata

    @staticmethod
    def _build_manifold_matrix(
        grouped_signal: Mapping[str, np.ndarray],
        neuron_ids: np.ndarray,
        time_indices: np.ndarray,
        average_trials: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        blocks = []
        sample_groups = []
        sample_trials = []
        sample_time = []

        for label, signal in grouped_signal.items():
            if signal.shape[0] == 0:
                continue

            if average_trials:
                mean_signal = np.nanmean(signal[:, time_indices, :], axis=0)
                blocks.append(mean_signal)
                sample_groups.extend([label] * mean_signal.shape[0])
                sample_trials.extend([-1] * mean_signal.shape[0])
                sample_time.extend(time_indices)
                continue

            selected_signal = signal[:, time_indices, :]
            block = selected_signal.reshape(-1, selected_signal.shape[-1])
            blocks.append(block)
            for trial_index in range(selected_signal.shape[0]):
                sample_groups.extend([label] * selected_signal.shape[1])
                sample_trials.extend([trial_index] * selected_signal.shape[1])
                sample_time.extend(time_indices)

        if not blocks:
            raise ValueError("No signal remained after trial grouping.")

        matrix = np.concatenate(blocks, axis=0)
        valid_samples = np.isfinite(matrix).all(axis=1)
        matrix = matrix[valid_samples]

        sample_metadata = {
            "group": np.asarray(sample_groups, dtype=object)[valid_samples],
            "trial_index": np.asarray(sample_trials, dtype=int)[valid_samples],
            "time_index": np.asarray(sample_time, dtype=int)[valid_samples],
        }
        feature_metadata = {
            "neuron_id": neuron_ids,
        }
        return matrix, sample_metadata, feature_metadata

    @staticmethod
    def _build_trial_average_tensor(
        grouped_signal: Mapping[str, np.ndarray],
        neuron_ids: np.ndarray,
        time_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        trial_labels = []
        tensors = []

        for label, signal in grouped_signal.items():
            if signal.shape[0] == 0:
                continue
            mean_signal = np.nanmean(signal[:, time_indices, :], axis=0)
            trial_labels.append(label)
            tensors.append(np.moveaxis(mean_signal, 1, 0)[:, :, None])

        if not tensors:
            raise ValueError("No signal remained after trial grouping.")

        tensor = np.concatenate(tensors, axis=2)
        valid_neurons = np.isfinite(tensor).all(axis=(1, 2))
        tensor = tensor[valid_neurons]
        metadata = {
            "neuron_id": neuron_ids[valid_neurons],
            "trial_label": np.asarray(trial_labels, dtype=object),
            "time_index": np.asarray(time_indices, dtype=int),
        }
        return tensor, metadata

    @staticmethod
    def _fit_pca(
        matrix: np.ndarray,
        n_components: int | float | None,
    ) -> tuple[PCA, np.ndarray]:
        if matrix.ndim != 2:
            raise ValueError(f"PCA expects a 2D matrix, got {matrix.shape}.")
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError(f"PCA requires at least a 2x2 matrix, got {matrix.shape}.")

        pca = PCA(n_components=_resolve_n_components(n_components, matrix.shape))
        scores = pca.fit_transform(matrix)
        return pca, scores

    def fit_session_pca(
        self,
        mode: PopulationMode,
        *,
        trial_groups: Mapping[str, Iterable[int]] | None = None,
        neuron_indices: np.ndarray | Iterable[int] | None = None,
        session_indices: Iterable[int] | None = None,
        time_window: tuple[int, int] | slice | Iterable[int] | None = None,
        average_trials: bool | None = None,
        n_components: int | float | None = 3,
    ) -> OrderedDict[str, SessionPCAResult]:
        if average_trials is None:
            average_trials = False
        if time_window is None:
            time_window = self.fit_time_window

        results: OrderedDict[str, SessionPCAResult] = OrderedDict()

        for session in self.iter_sessions(session_indices=session_indices):
            session_signal, session_neuron_ids = self._select_neurons(
                session,
                neuron_indices=neuron_indices,
            )
            time_indices = self._resolve_time_indices(
                session_signal.shape[1],
                time_window=time_window,
            )
            session_trial_groups = normalize_trial_groups(
                trial_groups,
                session.trial_types,
                trial_label_map=self.trial_label_map,
            )
            grouped_signal = self._group_session_signal(
                session_signal,
                session.trial_types,
                session_trial_groups,
            )

            if mode == "neuronal_embeddings":
                matrix, sample_metadata, feature_metadata = self._build_embedding_matrix(
                    grouped_signal,
                    neuron_ids=session_neuron_ids,
                    time_indices=time_indices,
                    average_trials=average_trials,
                )
            else:
                matrix, sample_metadata, feature_metadata = self._build_manifold_matrix(
                    grouped_signal,
                    neuron_ids=session_neuron_ids,
                    time_indices=time_indices,
                    average_trials=average_trials,
                )

            pca, scores = self._fit_pca(matrix, n_components=n_components)
            results[session.session_name] = SessionPCAResult(
                mode=mode,
                group=session.group,
                session_index=session.session_index,
                session_name=session.session_name,
                matrix=matrix,
                scores=scores,
                components=pca.components_,
                explained_variance_ratio=pca.explained_variance_ratio_,
                sample_metadata=sample_metadata,
                feature_metadata=feature_metadata,
                pca=pca,
            )

        return results

    def project_average_trajectories(
        self,
        session_result: SessionPCAResult,
        *,
        trial_groups: Mapping[str, Iterable[int]] | None = None,
        neuron_indices: np.ndarray | Iterable[int] | None = None,
    ) -> OrderedDict[str, np.ndarray]:
        session = self.iter_sessions(session_indices=[session_result.session_index])[0]
        session_signal, _ = self._select_neurons(
            session,
            neuron_indices=neuron_indices,
        )
        session_trial_groups = normalize_trial_groups(
            trial_groups,
            session.trial_types,
            trial_label_map=self.trial_label_map,
        )
        grouped_signal = self._group_session_signal(
            session_signal,
            session.trial_types,
            session_trial_groups,
        )

        trajectories = OrderedDict()
        for label, signal in grouped_signal.items():
            if signal.shape[0] == 0:
                continue
            mean_signal = np.nanmean(signal, axis=0)
            valid_timepoints = np.isfinite(mean_signal).all(axis=1)
            if not np.any(valid_timepoints):
                continue
            trajectories[label] = session_result.pca.transform(mean_signal[valid_timepoints])
        return trajectories

    def build_session_trial_average_tensor(
        self,
        *,
        trial_groups: Mapping[str, Iterable[int]] | None = None,
        neuron_indices: np.ndarray | Iterable[int] | None = None,
        session_index: int,
        time_window: tuple[int, int] | slice | Iterable[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if time_window is None:
            time_window = self.fit_time_window

        session = self.iter_sessions(session_indices=[session_index])[0]
        session_signal, session_neuron_ids = self._select_neurons(
            session,
            neuron_indices=neuron_indices,
        )
        time_indices = self._resolve_time_indices(
            session_signal.shape[1],
            time_window=time_window,
        )
        session_trial_groups = normalize_trial_groups(
            trial_groups,
            session.trial_types,
            trial_label_map=self.trial_label_map,
        )
        grouped_signal = self._group_session_signal(
            session_signal,
            session.trial_types,
            session_trial_groups,
        )
        return self._build_trial_average_tensor(
            grouped_signal,
            neuron_ids=session_neuron_ids,
            time_indices=time_indices,
        )

    def neuronal_embeddings(
        self,
        **kwargs,
    ) -> OrderedDict[str, SessionPCAResult]:
        return self.fit_session_pca("neuronal_embeddings", **kwargs)

    def neural_manifold(
        self,
        **kwargs,
    ) -> OrderedDict[str, SessionPCAResult]:
        return self.fit_session_pca("neural_manifold", **kwargs)
