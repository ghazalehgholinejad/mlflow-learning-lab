"""Project 3: register a model and promote it through a mutable alias."""

import argparse
from pathlib import Path

import mlflow
from sklearn.ensemble import RandomForestClassifier

from mlflow_lab import train_and_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--model-name", default="cancer-risk-classifier")
    parser.add_argument("--promote", action="store_true", help="Move the champion alias to this version")
    args = parser.parse_args()

    params = {"n_estimators": 160, "max_depth": 7, "random_state": 42}
    result = train_and_log(
        RandomForestClassifier(**params),
        "registry-random-forest",
        params,
        args.tracking_uri,
        "03-model-registry",
        Path("reports"),
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    version = mlflow.register_model(result["model_uri"], args.model_name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(args.model_name, "candidate", version.version)
    client.set_model_version_tag(args.model_name, version.version, "validation_status", "pending")
    if args.promote:
        client.set_registered_model_alias(args.model_name, "champion", version.version)
        client.set_model_version_tag(args.model_name, version.version, "validation_status", "approved")
    print(f"Registered {args.model_name} v{version.version} as candidate")


if __name__ == "__main__":
    main()

