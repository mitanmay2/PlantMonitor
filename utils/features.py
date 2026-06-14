"""Traditional computer-vision feature extraction."""

from __future__ import annotations

import cv2
import numpy as np

HISTOGRAM_BINS = 8

FEATURE_NAMES = [
    "mean_red",
    "mean_green",
    "mean_blue",
    "mean_hue",
    "mean_saturation",
    "mean_value",
]
FEATURE_NAMES += [
    f"hist_{channel}_{index}"
    for channel in ("red", "green", "blue")
    for index in range(HISTOGRAM_BINS)
]
FEATURE_NAMES += [
    "gray_mean",
    "gray_std",
    "gray_entropy",
    "laplacian_variance",
    "edge_density",
]


def _normalized_histogram(channel: np.ndarray) -> np.ndarray:
    histogram = cv2.calcHist(
        [channel], [0], None, [HISTOGRAM_BINS], [0, 256]
    ).flatten()
    total = histogram.sum()
    return histogram / total if total else histogram


def _entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    probabilities = histogram / max(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_features(
    image_bgr: np.ndarray, image_rgb: np.ndarray, image_hsv: np.ndarray
) -> np.ndarray:
    """Extract color, histogram, texture, sharpness, and edge features."""
    mean_rgb = image_rgb.mean(axis=(0, 1))
    mean_hsv = image_hsv.mean(axis=(0, 1))

    histograms = np.concatenate(
        [_normalized_histogram(image_rgb[:, :, channel]) for channel in range(3)]
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    texture = np.array(
        [
            gray.mean(),
            gray.std(),
            _entropy(gray),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
            np.count_nonzero(edges) / edges.size,
        ],
        dtype=np.float64,
    )

    features = np.concatenate([mean_rgb, mean_hsv, histograms, texture])
    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("Feature extraction produced an unexpected vector size.")
    return features.astype(np.float64)


def features_as_dict(values: np.ndarray) -> dict[str, float]:
    """Return a labeled feature dictionary for display and export."""
    return dict(zip(FEATURE_NAMES, values.astype(float)))
