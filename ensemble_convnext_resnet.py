"""Explicit two-stage probability ensemble: ConvNeXt first, then ResNet.

    p_convnext = (p_224 + p_320) / 2
    p_final = (p_convnext + p_resnet) / 2

Final weights: 25% ConvNeXt-224, 25% ConvNeXt-320, 50% ResNet.
Inputs must be probabilities before thresholding, with the same label order.
The teammate's binary submission (6).csv is NOT a probability export.

This CLI writes probabilities, not a ready submission. Calibrate final thresholds
on common held-out predictions in cnn_2_ensemble_resnet.ipynb first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def _validate_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(LABEL_COLS):
        raise ValueError(f"{name}: expected a nonempty N x {len(LABEL_COLS)} probability array.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}: probabilities must be finite and in [0, 1].")
    return values


def combine_probabilities(
    p_224: np.ndarray, p_320: np.ndarray, p_resnet: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ConvNeXt mean, final mean) for already aligned image rows.

    Reject an entirely binary ResNet array so a submission is not silently used
    as a confidence estimate. The same rule is applied to validation and test.
    """
    p_224 = _validate_probabilities(p_224, "ConvNeXt-224")
    p_320 = _validate_probabilities(p_320, "ConvNeXt-320")
    p_resnet = _validate_probabilities(p_resnet, "ResNet")
    if not (p_224.shape == p_320.shape == p_resnet.shape):
        raise ValueError("All models must have the same number of aligned image rows.")
    if np.isin(p_resnet, [0.0, 1.0]).all():
        raise ValueError(
            "ResNet contains only 0/1 labels. Supply raw sigmoid probabilities, "
            "not submission.csv. Use the archived hard-voting experiment for binary labels."
        )

    # Stage 1: our two resolution ensembles form ONE ConvNeXt branch.
    p_convnext = (p_224 + p_320) / 2.0

    # Stage 2: give the ConvNeXt branch and ResNet equal weight.
    p_final = (p_convnext + p_resnet) / 2.0

    return p_convnext, p_final


def read_probability_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str})
    if set(frame.columns) != {"image_id", *LABEL_COLS}:
        raise ValueError(f"{path}: expected image_id and exactly {LABEL_COLS}.")
    if frame.empty or frame.image_id.isna().any() or not frame.image_id.is_unique:
        raise ValueError(f"{path}: image_id must be nonempty, unique and nonmissing.")
    _validate_probabilities(frame[LABEL_COLS].to_numpy(), str(path))
    return frame[["image_id", *LABEL_COLS]]


def combine_frames(
    convnext224: pd.DataFrame, convnext320: pd.DataFrame, resnet: pd.DataFrame
) -> pd.DataFrame:
    """Align models by image_id and preserve the ConvNeXt-224 input row order."""
    for name, frame in [("ConvNeXt-224", convnext224), ("ConvNeXt-320", convnext320), ("ResNet", resnet)]:
        if set(frame.columns) != {"image_id", *LABEL_COLS}:
            raise ValueError(f"{name}: unexpected columns.")
        if frame.empty or frame.image_id.isna().any() or not frame.image_id.is_unique:
            raise ValueError(f"{name}: duplicate, missing or empty image IDs.")
    ids = convnext224.image_id.tolist()
    for frame in [convnext320, resnet]:
        if set(frame.image_id) != set(ids):
            raise ValueError("Models predict different image_id sets.")
    aligned320 = convnext320.set_index("image_id").loc[ids]
    aligned_resnet = resnet.set_index("image_id").loc[ids]
    _, p_final = combine_probabilities(
        convnext224[LABEL_COLS].to_numpy(),
        aligned320[LABEL_COLS].to_numpy(),
        aligned_resnet[LABEL_COLS].to_numpy(),
    )
    result = pd.DataFrame(p_final, columns=LABEL_COLS)
    result.insert(0, "image_id", ids)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convnext224", type=Path, required=True)
    parser.add_argument("--convnext320", type=Path, required=True)
    parser.add_argument("--resnet", type=Path, required=True, help="Raw probabilities, not binary submission.")
    parser.add_argument("--output", type=Path, required=True, help="Output probability CSV; not a thresholded submission.")
    args = parser.parse_args(argv)
    result = combine_frames(
        read_probability_csv(args.convnext224),
        read_probability_csv(args.convnext320),
        read_probability_csv(args.resnet),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} probability rows to {args.output}")


if __name__ == "__main__":
    main()
