import unittest

import numpy as np

from src.Decoding import Decoder


class DecoderTests(unittest.TestCase):
    def test_population_aggregate_decoding_is_high_for_separable_signal(self):
        rng = np.random.default_rng(0)
        labels = np.repeat([0, 1], 40)
        signal = rng.normal(scale=0.2, size=(80, 6, 5))
        signal[labels == 1, :, :3] += 1.5

        decoder = Decoder(task="classification", cv=5, random_state=0)
        result = decoder.decode(
            signal=signal,
            labels=labels,
            neuron_mode="population",
            temporal_mode="aggregate",
        )

        self.assertEqual(result.scores.ndim, 1)
        self.assertGreater(result.mean_score, 0.95)

    def test_single_neuron_decoding_returns_one_score_per_neuron(self):
        rng = np.random.default_rng(1)
        labels = np.repeat([0, 1], 50)
        signal = rng.normal(scale=0.3, size=(100, 5, 4))
        signal[labels == 1, :, 0] += 2.0

        decoder = Decoder(task="classification", cv=5, random_state=1)
        result = decoder.decode(
            signal=signal,
            labels=labels,
            neuron_mode="single-neuron",
            temporal_mode="aggregate",
        )

        self.assertEqual(result.scores.shape[0], 4)
        self.assertGreater(result.mean_score[0], 0.95)
        self.assertLess(np.nanmax(result.mean_score[1:]), 0.8)

    def test_temporal_decoding_accepts_time_varying_labels(self):
        rng = np.random.default_rng(2)
        n_trials = 80
        n_time = 4
        labels = np.tile(np.repeat([0, 1], n_trials // 2)[:, None], (1, n_time))
        signal = rng.normal(scale=0.25, size=(n_trials, n_time, 3))

        for time_idx in range(n_time):
            signal[labels[:, time_idx] == 1, time_idx, :] += 1.2

        decoder = Decoder(task="classification", cv=5, random_state=2)
        result = decoder.decode(
            signal=signal,
            labels=labels,
            neuron_mode="population",
            temporal_mode="per-timepoint",
        )

        self.assertEqual(result.scores.shape[0], n_time)
        self.assertTrue(np.all(result.mean_score > 0.9))

    def test_equal_neuron_subsampling_returns_expected_row_counts(self):
        rng = np.random.default_rng(3)
        labels = np.repeat([0, 1], 30)
        signal = rng.normal(scale=0.35, size=(60, 5, 8))
        signal[labels == 1, :, :] += 0.9

        decoder = Decoder(task="classification", cv=5, random_state=3)

        subsampling = decoder.equal_neuron_subsampling(
            signal=signal,
            labels=labels,
            neuron_groups={
                "small": np.arange(3),
                "large": np.arange(3, 8),
            },
            n_boot=4,
            neuron_mode="population",
            temporal_mode="aggregate",
        )
        self.assertEqual(len(subsampling), 8)
        self.assertEqual(set(subsampling["n_neurons"]), {3})


if __name__ == "__main__":
    unittest.main()
