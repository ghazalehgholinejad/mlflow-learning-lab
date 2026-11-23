"""Project 4: validate a candidate before moving the champion alias."""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier

from mlflow_lab import evaluate_promotion, train_and_log


def champion_metrics(client: mlflow.MlflowClient, model_name: str):
    """Return the current champion's metrics, or None on the first deployment."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    if not champion.run_id:
        return None
    return client.get_run(champion.run_id).data.metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--model-name", default="cancer-risk-classifier")
    parser.add_argument("--min-f1", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-f1-regression", type=float, default=0.01)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    params = {"n_estimators": 180, "max_depth": 8, "random_state": 42}
    result = train_and_log(
        RandomForestClassifier(**params),
        "quality-gated-random-forest",
        params,
        args.tracking_uri,
        "04-quality-gate",
        Path("reports"),
    )
    client = mlflow.MlflowClient(tracking_uri=args.tracking_uri)
    current_metrics = champion_metrics(client, args.model_name)
    report = evaluate_promotion(
        result["metrics"],
        current_metrics,
        min_f1=args.min_f1,
        min_recall=args.min_recall,
        max_f1_regression=args.max_f1_regression,
    )
    report_path = Path("reports/promotion-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    client.log_artifact(result["run_id"], str(report_path), "validation")
    client.set_tag(result["run_id"], "quality_gate", "passed" if report["approved"] else "failed")

    version = mlflow.register_model(result["model_uri"], args.model_name)
    client.set_registered_model_alias(args.model_name, "candidate", version.version)
    client.set_model_version_tag(
        args.model_name,
        version.version,
        "validation_status",
        "approved" if report["approved"] else "rejected",
    )
    if args.promote and report["approved"]:
        client.set_registered_model_alias(args.model_name, "champion", version.version)

    print(json.dumps(report, indent=2))
    if args.promote and not report["approved"]:
        raise SystemExit("Promotion blocked: the candidate failed the quality gate")
    if report["approved"] and not args.promote:
        print("Candidate passed. Re-run with --promote to move the champion alias.")


if __name__ == "__main__":
    main()
