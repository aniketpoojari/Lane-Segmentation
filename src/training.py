import argparse
import numpy as np
import torch
import mlflow
import itertools
from common import read_params
from load_and_augment import get_data_loader
from model import IoU, get_model, get_essentials


def run(train_loader, val_loader, model, optimizer, criterion, epoches):

    max_score = -1
    best_model = None
    for i in range(epoches):
        with torch.enable_grad():
            model.train()
            losses = np.empty(0)
            s = 0
            for X, y in train_loader:
                pred = model(X.cuda())
                loss = criterion(pred, y.cuda())
                losses = np.append(losses, [loss.item() * len(X)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                s += len(X)
            losses = losses.sum() / s

        with torch.no_grad():
            model.eval()
            scores = np.empty(0)
            for X, y in val_loader:
                pred = model(X.cuda())
                batch_scores = IoU(pred.cpu(), y)
                scores = np.append(scores, batch_scores)
            score = scores.mean()

            # print(
            #     "Epoch: {} / {} - IOU: {} - Loss: {} ".format(
            #         i + 1, epoches, scores, losses
            #     )
            # )

        if score > max_score:
            max_score = score
            best_model = model

    return best_model, max_score


def training(config_path):
    config = read_params(config_path)
    data_path = config["data"]
    x_train_dir = data_path["train_folder"]
    y_train_dir = data_path["trainannot_folder"]
    x_val_dir = data_path["test_folder"]
    y_val_dir = data_path["testannot_folder"]

    loader_path = config["loader"]
    resize = loader_path["resize"]
    batch_size = loader_path["batch_size"]
    num_workers = loader_path["num_workers"]

    training_path = config["training"]
    epoches = training_path["epoches"]
    start_channel = training_path["start_channel"]
    num_layers = training_path["num_layers"]
    lr = training_path["lr"]

    mlflow_config = config["mlflow_config"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    registered_model_name = mlflow_config["registered_model_name"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except:
        # print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name) as mlflow_run_outer:

        dic = {
            "resize": resize,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "start_channel": start_channel,
            "num_layers": num_layers,
            "lr": lr,
            "epoches": epoches,
        }

        combinations = itertools.product(*list(dic.values()))

        max_score = -2
        best_model = None
        best_params = None

        for i in combinations:
            with mlflow.start_run(nested=True, run_name="asdf") as mlflow_run:

                params = dict()
                for j, p in enumerate(dic.keys()):
                    params[p] = i[j]

                train_loader_args = {
                    "X_dir": x_train_dir,
                    "y_dir": y_train_dir,
                    "resize": params["resize"],
                    "batch_size": params["batch_size"],
                    "num_workers": params["num_workers"],
                }
                val_loader_args = {
                    "X_dir": x_val_dir,
                    "y_dir": y_val_dir,
                    "resize": params["resize"],
                    "batch_size": params["batch_size"],
                    "num_workers": params["num_workers"],
                }
                train_loader = get_data_loader(**train_loader_args)
                val_loader = get_data_loader(**val_loader_args)

                model_args = {
                    "start_channel": params["start_channel"],
                    "num_layers": params["num_layers"],
                }
                model = get_model(**model_args)

                get_essentials_args = {"model": model, "lr": params["lr"]}
                criterion, optimizer = get_essentials(**get_essentials_args)

                run_args = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "model": model,
                    "optimizer": optimizer,
                    "criterion": criterion,
                    "epoches": params["epoches"],
                }
                try:
                    model, score = run(**run_args)
                except:
                    continue

                mlflow.log_params(params)
                mlflow.log_metric("IOU", score)
                mlflow.pytorch.log_model(model, f"{mlflow_run.info.run_id}")

                if score > max_score:
                    max_score = score
                    best_model = model
                    best_params = params

        mlflow.log_params(best_params)
        mlflow.log_metric("IOU", max_score)
        mlflow.pytorch.log_model(
            best_model,
            f"{mlflow_run_outer.info.run_id}",
            registered_model_name=registered_model_name,
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
