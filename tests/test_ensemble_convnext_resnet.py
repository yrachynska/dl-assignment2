"""CPU checks for explicit weights, ID alignment and invalid-input rejection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_convnext_resnet import LABEL_COLS, combine_frames, combine_probabilities, main


class EnsembleTests(unittest.TestCase):
    def test_two_stage_weights_and_branch_mean(self):
        p224 = np.array([[0.8, 0.1, 0.4, 0.7, 0.3], [0.2, 0.9, 0.6, 0.1, 0.8]])
        p320 = np.array([[0.4, 0.3, 0.8, 0.5, 0.1], [0.6, 0.7, 0.2, 0.3, 0.4]])
        resnet = np.array([[0.2, 0.6, 0.9, 0.3, 0.8], [0.8, 0.2, 0.1, 0.7, 0.4]])
        branch, final = combine_probabilities(p224, p320, resnet)
        np.testing.assert_allclose(branch, [[0.6, 0.2, 0.6, 0.6, 0.2], [0.4, 0.8, 0.4, 0.2, 0.6]])
        np.testing.assert_allclose(final, 0.25 * p224 + 0.25 * p320 + 0.5 * resnet)
        self.assertAlmostEqual(final[0, 0], 0.4)

    def test_shuffled_ids_and_cli_output(self):
        frame = pd.DataFrame(np.full((3, 5), 0.2), columns=LABEL_COLS)
        frame.insert(0, "image_id", ["0003", "0001", "0002"])
        second = frame.copy()
        second[LABEL_COLS] = 0.6
        resnet = frame.copy()
        resnet[LABEL_COLS] = [[0.1] * 5, [0.5] * 5, [0.9] * 5]
        expected = combine_frames(frame, second, resnet)
        shuffled = combine_frames(frame, second.iloc[::-1], resnet.iloc[[1, 2, 0]])
        pd.testing.assert_frame_equal(expected, shuffled)
        self.assertEqual(expected.image_id.tolist(), ["0003", "0001", "0002"])
        np.testing.assert_allclose(expected[LABEL_COLS], [[0.25] * 5, [0.45] * 5, [0.65] * 5])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name, values in [("224", frame), ("320", second.iloc[::-1]), ("resnet", resnet.iloc[::-1])]:
                values.to_csv(folder / f"{name}.csv", index=False)
            output = folder / "output" / "probabilities.csv"
            main(["--convnext224", str(folder / "224.csv"), "--convnext320", str(folder / "320.csv"),
                  "--resnet", str(folder / "resnet.csv"), "--output", str(output)])
            actual = pd.read_csv(output, dtype={"image_id": str})
            pd.testing.assert_frame_equal(expected, actual)

    def test_binary_resnet_is_not_a_probability_export(self):
        probabilities = np.full((2, 5), 0.4)
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            combine_probabilities(probabilities, probabilities, np.array([[0, 1, 0, 1, 0]] * 2))

    def test_invalid_shapes_values_and_ids(self):
        good = np.full((2, 5), 0.4)
        for bad in [np.full((2, 4), 0.4), np.full((1, 5), 0.4), np.full((2, 5), np.nan), np.full((2, 5), 1.1)]:
            with self.assertRaises(ValueError):
                combine_probabilities(good, bad, good)
        frame = pd.DataFrame(good, columns=LABEL_COLS)
        frame.insert(0, "image_id", ["a", "b"])
        duplicate = frame.copy()
        duplicate["image_id"] = ["a", "a"]
        with self.assertRaises(ValueError):
            combine_frames(frame, duplicate, frame)
        missing = frame.copy()
        missing["image_id"] = ["a", "c"]
        with self.assertRaises(ValueError):
            combine_frames(frame, frame, missing)


if __name__ == "__main__":
    unittest.main()
