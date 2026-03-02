from common import read_params
import argparse
import mlflow

# from mlflow.tracking import MlflowClient
# from pprint import pprint
# import shutil
import pandas as pd
import torch

import shutil


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"]
    experiment_name = mlflow_config["experiment_name"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # shifting best model to websitefolder
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment_id))
    # df = df[df["tags.mlflow.parentRunId"].isnull()]
    df = df[df["metrics.IOU"] == df["metrics.IOU"].max()]
    run_id = df["run_id"].values
    artifact_uri = df["artifact_uri"].values
    src = f"{artifact_uri[0]}/{run_id[0]}-model/data/model.pth"
    dest = "saved_models/model.pth"
    shutil.copyfile(src, dest)

    # setting best model to production
    """runs = mlflow.search_runs(experiment_ids=experiment_id)
    lowest = runs["metrics.r2_test"].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs["metrics.r2_test"] == lowest]["run_id"][0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            address = mv["source"]
            # pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Staging"
            )
    address += "\model.cb"

    model_path = config["webapp_model_dir"]

    shutil.copyfile(address, model_path)"""


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
