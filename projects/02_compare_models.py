"""Project 2: compare candidate models through logged evidence."""

import argparse
from pathlib import Path

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlflow_lab import train_and_log
from mlflow_lab.core import write_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    candidates = [
        (
            "logistic-regression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
            {"max_iter": 1000, "random_state": 42},
        ),
        (
            "random-forest",
            RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42),
            {"n_estimators": 120, "max_depth": 6, "random_state": 42},
        ),
    ]
    rows = []
    for name, model, params in candidates:
        result = train_and_log(
            model, name, params, args.tracking_uri, "02-model-comparison", Path("reports")
        )
        rows.append({"model": name, "run_id": result["run_id"], **result["metrics"]})

    leaderboard = write_leaderboard(rows, Path("reports/leaderboard.csv"))
    mlflow.set_tracking_uri(args.tracking_uri)
    best_run_id = str(leaderboard.iloc[0]["run_id"])
    client = mlflow.MlflowClient()
    client.set_tag(best_run_id, "selection_status", "best-by-f1")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

