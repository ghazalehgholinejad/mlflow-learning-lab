# MLOps and MLflow — practical course

## 1. What problem does MLOps solve?

A notebook may prove that a model can work. A dependable product must repeatedly transform data into a tested, traceable, deployable model. **MLOps** applies software and operations practices to that lifecycle. Its concerns include versioning, reproducibility, automated tests, delivery, monitoring, governance, and feedback after deployment.

MLflow does not replace all of MLOps. It supports several important parts:

- **Tracking:** records runs, parameters, metrics, tags, and artifacts.
- **Models:** packages a model with dependencies and an input/output contract.
- **Registry:** versions models and attaches aliases, tags, and descriptions.
- **Serving and integrations:** help move a packaged model toward inference.

Git versions source code; MLflow tracks executions and model artifacts. You normally need both.

## 2. The vocabulary

- **Experiment:** a container for related runs.
- **Run:** one execution of training or evaluation.
- **Parameter:** an input chosen before training, such as `max_depth`.
- **Metric:** a measured result, such as F1.
- **Artifact:** an output file, such as a plot or model.
- **Signature:** the model input/output schema.
- **Registered model:** a named collection of model versions.
- **Alias:** a movable name that points to a version, such as `champion`.

## 3. Project 01 — tracking

Run `python projects/01_track_experiment.py`. The script makes the data split deterministic, records the hyperparameters, logs four metrics and a confusion matrix, and packages the trained pipeline. Change `C` and rerun. In the UI, compare runs and ask whether the observed difference is large enough to matter.

Parameters and metrics are different: a parameter explains what you chose; a metric describes what happened. Tags add searchable context but are not numeric time series. Artifacts hold larger outputs.

## 4. Project 02 — comparison

Run `python projects/02_compare_models.py`. Two candidates see exactly the same train/test split. The script sorts by F1 and then accuracy, writes a leaderboard, and tags the selected run. This is a transparent policy, but not automatically a good production policy. Real selection should also consider latency, calibration, subgroup performance, cost, and uncertainty.

Never tune repeatedly against a final test set. Use validation or cross-validation for selection, and reserve a test set for the final estimate.

## 5. Project 03 — registry

Run `python projects/03_registry_aliases.py`. It registers a model version, assigns `candidate`, and marks validation as pending. Running it with `--promote` also moves `champion` to that version and marks it approved.

An alias is useful because inference code can load `models:/cancer-risk-classifier@champion` while promotion changes the target centrally. Promotion should follow checks; this lab makes the decision explicit but does not pretend to implement a full approval system.

## 6. Project 04 — automated quality gate

Run `python projects/04_quality_gate.py` to train and register a candidate without changing the production alias. The policy checks absolute F1 and recall thresholds. When a champion already exists, it also blocks an unacceptable F1 regression. The complete decision is logged as a JSON artifact, so a reviewer can see why the candidate passed or failed.

Run again with `--promote`. The `champion` alias moves only when every check passes. Try `--min-f1 0.99 --promote` to observe a blocked release. This pattern turns promotion from an informal choice into a testable policy. A real system would add latency, fairness, data-quality and security checks and would run the gate in a protected deployment workflow.

## 7. Local versus team architecture

This lab uses SQLite and local artifacts. It is simple and reproducible for one learner. A team commonly runs a tracking server, stores metadata in a managed database, stores artifacts in object storage, and protects the service with authentication and network controls.

## 8. Common mistakes

- Logging a score without the dataset/version that produced it.
- Logging a model without a signature and input example.
- Comparing runs produced by different splits without recording that difference.
- Treating the best validation score as proof of production quality.
- Committing `mlflow.db`, `mlruns`, or model artifacts to Git.
- Moving a production alias without validation and rollback criteria.

## 9. Exercises

1. Add ROC AUC and explain when it is more useful than accuracy.
2. Add a third candidate model and keep the comparison deterministic.
3. Add latency to the leaderboard and define a multi-objective policy.
4. Load `models:/cancer-risk-classifier@champion` and make predictions.
5. Add a test that rejects promotion below a chosen F1 threshold.
6. Replace local artifacts with object storage in a separate deployment branch.

## 10. Completion checklist

You understand the lab when you can explain every logged field, reproduce a run, justify a selection policy, load a model by alias, and describe what is still missing for production.
