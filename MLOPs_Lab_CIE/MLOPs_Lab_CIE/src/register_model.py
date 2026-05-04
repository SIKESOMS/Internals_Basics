def get_best_run():
    """Return the top-level run with the lowest rmse that has a logged model artifact."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        order_by=["metrics.rmse ASC"],
    )

    for run in runs:
        if "rmse" not in run.data.metrics:
            continue
        # skip nested runs (they have a parentRunId tag)
        if run.data.tags.get("mlflow.parentRunId"):
            continue
        artifacts = client.list_artifacts(run.info.run_id)
        if any(a.is_dir for a in artifacts):
            return run

    raise RuntimeError("No top-level runs with a model artifact found.")