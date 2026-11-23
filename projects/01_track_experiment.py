"""Project 1: manually track one reproducible training run."""

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlflow_lab import train_and_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
    model = make_pipeline(StandardScaler(), LogisticRegression(**params))
    result = train_and_log(
        model=model,
        model_name="logistic-regression",
        params=params,
        tracking_uri=args.tracking_uri,
        experiment_name="01-manual-tracking",
        output_dir=Path("reports"),
    )
    print(f"Run: {result['run_id']} | F1: {result['metrics']['f1']:.3f}")


if __name__ == "__main__":
    main()

