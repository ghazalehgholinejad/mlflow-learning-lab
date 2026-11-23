"""Small, testable building blocks shared by all exercises."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def configure_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at a backend and create/select an experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def load_data(test_size: float = 0.25, random_state: int = 42):
    """Return a deterministic, public binary-classification dataset split."""
    dataset = load_breast_cancer(as_frame=True)
    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )


def train_and_log(
    model: Any,
    model_name: str,
    params: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Train a model, log reproducibility metadata, and return its run summary."""
    configure_tracking(tracking_uri, experiment_name)
    x_train, x_test, y_train, y_test = load_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "task": "binary-classification",
                "dataset": "sklearn-breast-cancer",
                "model_family": model_name,
            }
        )

        figure_path = output_dir / f"{run.info.run_id}-confusion-matrix.png"
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=140)
        plt.close()
        mlflow.log_artifact(str(figure_path), artifact_path="evaluation")

        input_example = x_test.iloc[:3]
        signature = infer_signature(x_test, predictions)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

    return {
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "metrics": metrics,
        "input_example": input_example,
    }


def write_leaderboard(rows: list[dict[str, Any]], destination: Path) -> pd.DataFrame:
    """Write a stable, human-readable model comparison table."""
    frame = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame
