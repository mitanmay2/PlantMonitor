"""Model training, persistence, prediction, and evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.features import FEATURE_NAMES

MODEL_NAMES = ("Random Forest", "SVM", "Logistic Regression")
LABEL_COLUMN = "label"
RANDOM_STATE = 42


def create_synthetic_dataset(samples: int = 900) -> pd.DataFrame:
    """Create a reproducible fallback dataset in the extracted feature space.

    This lets the application run immediately. For real-world accuracy, replace
    it by uploading features extracted from labeled plant images.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    labels = rng.integers(0, 2, size=samples)
    rows = []

    for diseased in labels:
        green = rng.normal(145 if not diseased else 92, 20)
        red = rng.normal(75 if not diseased else 125, 22)
        blue = rng.normal(62 if not diseased else 72, 18)
        hue = rng.normal(55 if not diseased else 30, 10)
        saturation = rng.normal(135 if not diseased else 155, 25)
        value = rng.normal(155 if not diseased else 140, 22)

        hist = []
        for channel_peak in (
            red / 255 * 7,
            green / 255 * 7,
            blue / 255 * 7,
        ):
            bins = np.arange(8)
            distribution = np.exp(-0.5 * ((bins - channel_peak) / 1.4) ** 2)
            distribution += rng.uniform(0.01, 0.08, size=8)
            hist.extend(distribution / distribution.sum())

        gray_mean = 0.299 * red + 0.587 * green + 0.114 * blue
        gray_std = rng.normal(31 if not diseased else 48, 8)
        entropy = rng.normal(6.2 if not diseased else 7.0, 0.45)
        laplacian = rng.lognormal(4.0 if not diseased else 4.6, 0.5)
        edge_density = rng.normal(0.10 if not diseased else 0.19, 0.04)

        row = [
            red,
            green,
            blue,
            hue,
            saturation,
            value,
            *hist,
            gray_mean,
            gray_std,
            entropy,
            laplacian,
            edge_density,
        ]
        rows.append(row)

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame[FEATURE_NAMES] = frame[FEATURE_NAMES].clip(lower=0)
    frame[LABEL_COLUMN] = labels
    return frame


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a custom feature CSV."""
    missing = [column for column in FEATURE_NAMES if column not in frame.columns]
    if LABEL_COLUMN not in frame.columns:
        missing.append(LABEL_COLUMN)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    clean = frame[FEATURE_NAMES + [LABEL_COLUMN]].copy()
    clean[FEATURE_NAMES] = clean[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    clean[LABEL_COLUMN] = clean[LABEL_COLUMN].replace(
        {
            "Healthy": 0,
            "healthy": 0,
            "Diseased": 1,
            "diseased": 1,
        }
    )
    clean[LABEL_COLUMN] = pd.to_numeric(clean[LABEL_COLUMN], errors="coerce")
    clean = clean.dropna()
    clean[LABEL_COLUMN] = clean[LABEL_COLUMN].astype(int)

    if len(clean) < 20:
        raise ValueError("The dataset must contain at least 20 valid rows.")
    if set(clean[LABEL_COLUMN].unique()) != {0, 1}:
        raise ValueError("The label column must contain both 0 (Healthy) and 1 (Diseased).")
    return clean


def build_model(model_name: str) -> Any:
    """Build one of the supported scikit-learn classifiers."""
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "SVM":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def train_and_evaluate(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Train a model and return a serializable model bundle."""
    clean = validate_dataset(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        clean[FEATURE_NAMES],
        clean[LABEL_COLUMN],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=clean[LABEL_COLUMN],
    )
    model = build_model(model_name)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }
    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "feature_names": FEATURE_NAMES,
        "training_rows": len(clean),
        "class_distribution": clean[LABEL_COLUMN].value_counts().to_dict(),
        "source": "custom" if len(clean) != 900 else "synthetic fallback",
    }


def model_path(models_dir: Path, model_name: str) -> Path:
    slug = model_name.lower().replace(" ", "_")
    return models_dir / f"{slug}.pkl"


def save_bundle(bundle: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_or_train_model(models_dir: Path, model_name: str) -> dict[str, Any]:
    """Load a persisted model or train a fallback model automatically."""
    destination = model_path(models_dir, model_name)
    if destination.exists():
        try:
            with destination.open("rb") as handle:
                bundle = pickle.load(handle)
            if bundle.get("feature_names") == FEATURE_NAMES:
                return bundle
        except (OSError, pickle.PickleError, EOFError):
            pass

    bundle = train_and_evaluate(create_synthetic_dataset(), model_name)
    save_bundle(bundle, destination)
    return bundle


def predict_leaf(bundle: dict[str, Any], features: np.ndarray) -> dict[str, Any]:
    """Predict leaf health and return dashboard-ready fields."""
    feature_frame = pd.DataFrame([features], columns=FEATURE_NAMES)
    model = bundle["model"]
    predicted_class = int(model.predict(feature_frame)[0])
    probabilities = model.predict_proba(feature_frame)[0]
    classes = list(model.classes_)
    probability_by_class = dict(zip(classes, probabilities))
    confidence = float(probability_by_class[predicted_class])
    healthy_probability = float(probability_by_class.get(0, 1.0 - confidence))

    if predicted_class == 0:
        prediction = "Healthy Leaf"
        status = "Healthy"
        recommendation = (
            "Continue regular watering, balanced sunlight exposure, and routine "
            "inspection for early signs of stress."
        )
    else:
        prediction = "Possible Leaf Disease"
        status = "Attention Needed"
        recommendation = (
            "Isolate the affected plant, remove severely damaged leaves, avoid "
            "overhead watering, and consult a local plant specialist for diagnosis."
        )

    return {
        "prediction": prediction,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "health_score": healthy_probability,
        "status": status,
        "recommendation": recommendation,
    }


def feature_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    """Return model-native feature influence where available."""
    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif isinstance(model, Pipeline):
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            values = np.abs(classifier.coef_[0])
        else:
            return pd.DataFrame(columns=["feature", "importance"])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    result = pd.DataFrame({"feature": FEATURE_NAMES, "importance": values})
    return result.sort_values("importance", ascending=False).head(12)
