from pathlib import Path

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlflow_lab import evaluate_promotion, load_data, train_and_log


def test_dataset_split_is_deterministic():
    first = load_data()
    second = load_data()
    assert first[0].equals(second[0])
    assert len(first[0]) + len(first[1]) == 569


def test_training_logs_metrics_model_and_artifact(tmp_path: Path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = train_and_log(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "test-model",
        {"max_iter": 1000, "random_state": 42},
        tracking_uri,
        "test-experiment",
        tmp_path / "reports",
    )
    run = mlflow.MlflowClient().get_run(result["run_id"])
    assert 0.0 <= run.data.metrics["f1"] <= 1.0
    assert run.data.tags["dataset"] == "sklearn-breast-cancer"
    assert mlflow.pyfunc.load_model(result["model_uri"]).predict(result["input_example"]).shape == (3,)


def test_quality_gate_approves_healthy_candidate():
    report = evaluate_promotion(
        {"f1": 0.94, "recall": 0.96},
        {"f1": 0.95, "recall": 0.95},
        max_f1_regression=0.01,
    )
    assert report["approved"] is True
    assert all(report["checks"].values())


def test_quality_gate_blocks_weak_candidate():
    report = evaluate_promotion(
        {"f1": 0.88, "recall": 0.89},
        {"f1": 0.95, "recall": 0.96},
    )
    assert report["approved"] is False
    assert report["checks"] == {
        "minimum_f1": False,
        "minimum_recall": False,
        "f1_regression_limit": False,
    }
