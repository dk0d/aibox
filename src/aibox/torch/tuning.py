import multiprocessing as mp
import shutil
from typing import Any

import numpy as np
import ray
import ray.train.lightning

# import ray.train as ray_train
# from ray.train.lightning import prepare_trainer
import torch
from omegaconf import OmegaConf
from ray import air, tune
from ray.air.config import CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from aibox.config import config_merge, config_update
from aibox.logger import get_logger

# import os
# from aibox.slurm.hpc import HyperOptWrapper, SlurmCluster
# from lightning import seed_everything
# from lightning.strategies.ddp import DDPStrategy
# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from aibox.torch.training import (
    build_cli_parser,
    train_and_test,
)
from aibox.utils import as_path, is_list, print

# os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

# from ax.plot.contour import plot_contour
# from ax.plot.trace import optimization_trace_single_method
# from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render, init_notebook_plotting
# from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN


LOGGER = get_logger(__name__)


def default_tune_func(tune_config, config):
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    # setup for config merge

    config_update(
        config,
        "trainer.callbacks.ray_train_checkpoint_report",
        dict(
            __classpath__="aibox.torch.training.RayTuneReportCheckpointCallback",
            # metrics={"loss": config.tuner.metric},
            on="validation_end",
            save_checkpoints=True,
        ),
    )
    config.trainer.update(enable_progress_bar=False)

    tune_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in tune_config.items()])
    train_config = config_merge(config, tune_config)

    # config_update(
    #     config,
    #     "trainer.callbacks.ray_train_report",
    #     dict(
    #         __classpath__="aibox.torch.training.RayTrainReportCallback",
    #     ),
    # )

    # mlflow.pytorch.autolog(log_models=True)
    results = train_and_test(train_config, should_init_logger=True)
    if isinstance(results, list):
        results = np.mean([r["test/loss"] for r in results])
    elif results is not None:
        results = results["test/loss"]
    return results


def get_tuner_type(_type, values, mode="ray"):
    """
    Create the search space for a given hyperparameter.
    Uses the `type` and `search_space` values from the config file.

    Supported Types:
        - `choice` (Ax, Ray)
        - `uniform` (Ax, Ray)
        - `loguniform` (Ax, Ray)
        - `randint` (Ax, Ray)
        - `randn` (Ax, Ray)
        - `quniform` (Ray)
        - `qloguniform` (Ray)
        - `qrandint` (Ray)
        - `qrandn` (Ray)
        - `grid_search` (Ray)
        - `rand_grid_search` (Ray)


    Attributes:
        _type: The type of the hyperparameter.
        values: The values of the hyperparameter.
        mode: The mode of the tuner. Either "ray" or "ax".
    """
    if mode == "ray":
        from ray import tune
    else:
        tune = None

    if is_list(values):
        values = list(OmegaConf.to_container(values))  # type: ignore

    match _type:
        case "choice":
            if mode == "ray":
                assert tune is not None
                return tune.choice(values)
            # from ax.core.parameter import ChoiceParameter
            return {
                "values": values,
                "type": "choice",
                "value_type": "int",
                "is_ordered": False,
            }
        case "uniform":
            if mode == "ray":
                assert tune is not None
                return tune.uniform(*values)  # type: ignore
            return {"bounds": values, "type": "range", "value_type": "float"}
        case "loguniform":
            if mode == "ray":
                assert tune is not None
                return tune.loguniform(*values)  # type: ignore
            return {"bounds": values, "type": "range", "log_scale": True, "value_type": "float"}
        case "randint":
            if mode == "ray":
                assert tune is not None
                return tune.randint(*values)  # type: ignore
            return {"bounds": values, "type": "range", "value_type": "int"}
        case "randn":
            if mode == "ray":
                assert tune is not None
                return tune.randn(*values)  # type: ignore
            return {"bounds": values, "type": "range", "value_type": "float"}
        case "quniform":
            if mode == "ray":
                assert tune is not None
                return tune.quniform(*values)  # type: ignore
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qloguniform":
            if mode == "ray":
                assert tune is not None
                return tune.qloguniform(*values)  # type: ignore
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qrandint":
            if mode == "ray":
                assert tune is not None
                return tune.qrandint(*values)  # type: ignore
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qrandn":
            if mode == "ray":
                assert tune is not None
                return tune.qrandn(*values)  # type: ignore
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "grid_search":
            if mode == "ray":
                assert tune is not None
                return tune.grid_search(values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        # case "rand_grid_search":
        #     if mode == "ray":
        #         assert tune is not None
        #         return tune.grid_search(values, randomize=True)
        #     raise NotImplementedError(f"{_type} not implemented for Ax")
        case _:
            raise ValueError(f"Unknown type: {_type}")


# def gather_search_spaces(config, mode="ray") -> dict[str, Any] | list[dict[str, Any]]:
#     """
#     Gather the search spaces for all hyperparameters given a configuration.
#
#     Attributes:
#         config: The configuration file.
#         mode: The mode of the tuner. Either "ray" or "ax".
#
#     Returns:
#         A dictionary of search spaces.
#     """
#     match mode:
#         case "ray":
#             return gather_search_spaces_ray(config)
#         case "ax":
#             tune_config = []
#             for s, v in config.tuner.search_spaces.items():
#                 for k, v in v.items():
#                     tk = ".".join([s, "args", k])
#                     c = get_tuner_type(v["type"], v["values"], mode="ax")
#                     c["name"] = tk
#                     tune_config.append(c)
#         case _:
#             raise ValueError(f"Unknown mode: {mode}")
#     return tune_config


def gather_search_spaces_ray(config) -> dict[str, Any]:
    """
    Gather the Ray search spaces for all hyperparameters given a configuration.

    Attributes:
        config: The configuration file.

    Returns:
        A dictionary of search spaces.
    """
    tune_config = {}
    for s, v in config.tuner.search_spaces.items():
        for k, v in v.items():
            tk = ".".join([s, "args", k])
            tune_config[tk] = get_tuner_type(v["type"], v["values"])
    return tune_config


class MLFlowTuneLogCallback(MLflowLoggerCallback):
    def __init__(
        self,
        config,
        tracking_uri: str | None = None,
        *,
        registry_uri: str | None = None,
        experiment_name: str | None = None,
        tags: dict | None = None,
        tracking_token: str | None = None,
        save_artifact: bool = False,
    ):
        super().__init__(
            tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            tags=tags,
            tracking_token=tracking_token,
            save_artifact=save_artifact,
        )
        self.config = config

    def setup(self, *args, **kwarg):
        super().setup(*args, **kwarg)
        # self.mlflow_util.log_params(self.config)
        if "tags" in self.config:
            self.tags = self.config.tags

    def log_dict(self, run_id, _dict, file_name):
        if run_id and self.mlflow_util._run_exists(run_id):
            client = self.mlflow_util._get_client()
            client.log_dict(run_id=run_id, dictionary=_dict, artifact_file=file_name)
        else:
            self.mlflow_util._mlflow.log_dict(dictionary=_dict, artifact_file=file_name)

    def log_trial_start(self, trial):
        # Create run if not already exists.
        if trial not in self._trial_runs:
            # Set trial name in tags
            tags = self.tags.copy()
            tags["trial_name"] = str(trial)

            run = self.mlflow_util.start_run(tags=tags, run_name=str(trial))
            self._trial_runs[trial] = run.info.run_id

        run_id = self._trial_runs[trial]
        trial_dir = trial.logdir
        if trial_dir is not None:
            # Write run_id to file.
            (as_path(trial_dir) / "run_id.txt").write_text(str(run_id))

        # LOGGER.info(f"Experiment Name: {self.experiment_name}")
        # LOGGER.info(f"Run ID: {self._trial_runs[trial]}")
        # LOGGER.info(f"Trial Name: {ray_train.get_context().get_trial_name()}")


def tune_ray(config, tune_fn=None):
    """
    Initialize Ray Tune and run hyperparameter tuning.

    Warning: RayTune support for hyperparameter tuning not fully tested yet
        Tested only on single, local machine
        See more at [MLFlow Ray Example](https://docs.ray.io/en/latest/tune/exmples/includes/mlflow_ptl_example.html)

    Attributes:
        config: A dictionary containing the configuration for the experiment.

    """

    ray.init()

    # from ray.tune.search.ax import AxSearch
    # from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

    # LOGGER.info(f"TUNE Logging to: {config.logging.tracking_uri}")
    param_space = gather_search_spaces_ray(config)
    tune_fn = tune_fn or default_tune_func
    trainable = tune.with_parameters(tune_fn, config=config)
    if shutil.which("sbatch") is not None:
        n_cpu = config.slurm.cpus_per_task
        n_gpu = config.slurm.ngpu
    else:
        n_cpu = mp.cpu_count()
        n_gpu = torch.cuda.device_count()

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": n_cpu, "gpu": n_gpu}),
        tune_config=tune.TuneConfig(
            metric=config.tuner.metric,
            mode=config.tuner.mode,
            num_samples=config.tuner.num_samples,
        ),
        run_config=air.RunConfig(
            name=config.name,
            storage_path=(as_path(config.project_root) / "logs/ray_results").as_posix(),
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=config.tuner.metric,
                checkpoint_score_order=config.tuner.mode,
            ),
            callbacks=[
                MLFlowTuneLogCallback(
                    config=config,
                    experiment_name=config.name,
                    tracking_uri=config.logging.tracking_uri,
                    save_artifact=True,
                ),
            ],
        ),
        param_space=param_space,
    )
    results = tuner.fit()
    return results


# def tune_ax(config):
#     """
#     Initialize Ax and run hyperparameter tuning.
#     Read more: [Ax Tune CNN Tutorial](https://ax.dev/tutorials/tune_cnn.html)
#
#     Note: Tuning Tags
#
#         Tuning also updates the tags of the runs so that all runs can be
#         grouped together and tracked / filtered in MLFlow.
#
#     Tags
#
#     - `tune_group_id`: A unique ID for the tuning group
#     - `tune_trial`: The trial number for the run
#
#     Attributes:
#         config: A dictionary containing the configuration for the experiment.
#
#     """
#
#     # from ax.plot.contour import plot_contour
#     # from ax.plot.trace import optimization_trace_single_method
#     # from ax.utils.notebook.plotting import init_notebook_plotting, render
#     from ax.service.managed_loop import optimize
#
#     tune_param = gather_search_spaces(config, mode="ax")
#
#     global trial
#     trial = 0
#     tune_group_id = uuid.uuid4().hex
#     OmegaConf.update(config, "tags.tune_group_id", tune_group_id)
#
#     def _wrapper(_tune_config):
#         global trial
#         trial += 1
#         OmegaConf.update(config, "tags.tune_trial", f"{trial}")
#         return tune_train(_tune_config, config)
#
#     best_params, values, experiment, model = optimize(
#         parameters=tune_params,
#         evaluation_function=_wrapper,
#         objective_name="loss",
#         minimize=True,
#         total_trials=3,
#     )
#
#     OmegaConf.update(config, "tags.tune_trial", "best")
#
#     tune_train(best_params, config)
#
#     LOGGER.info(f"Best: {best_params}, Values: {values}, Experiment: {experiment}")


# def tune_experiment(args=None):
#     """
#     Run hyperparameter tuning.
#
#     If `mode` is `None`, then the `mode` is read from the command line arguments.
#
#     Attributes:
#         mode: The mode of the tuner. Either "ray" or "ax".
#     """
#     if args is None:
#         mode_parser = argparse.ArgumentParser(add_help=False)
#         mode_parser.add_argument("--mode", type=str, default="ray", choices=["ray", "ax"])
#         known_args, other = mode_parser.parse_known_args()
#         mode = known_args.mode
#     else:
#         other = None
#
#     cli = build_cli_parser()
#
#     config = cli.parse_args(other)
#
#     if config.exp_name is None:
#         LOGGER.error("Experiment name must be specified")
#         sys.exit(1)
#
#     match mode:
#         case "ray":
#             tune_ray(config)
#         case "ax":
#             tune_ax(config)


# def tune_slurm(mode=None):
#     hyperparams = HyperOptWrapper()

# # init cluster
# cluster = SlurmCluster(hyperparam_optimizer=hyperparams, slurm_log_path="./", python_cmd="python")
#
# # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
# cluster.notify_job_status(email="some@email.com", on_done=True, on_fail=True)
#
# # set the job options. In this instance, we'll run 20 different models
# # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
# cluster.per_experiment_nb_gpus = 1
# cluster.per_experiment_nb_nodes = 1
#
# # we'll request 10GB of memory per node
# cluster.memory_mb_per_node = 10000
#
# # set a walltime of 10 minues
# cluster.job_time = "10:00"
#
# # 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
# # you must provide your own loading and saving function which the cluster object will call
# cluster.minutes_to_checkpoint_before_walltime = 1
#
# # run the models on the cluster
# cluster.optimize_parallel_cluster_gpu(train, nb_trials=20, job_name="first_tt_batch", job_display_name="my_batch")


def main(args=None):
    """
    Main entry-point for tuning a model.
    """
    cli = build_cli_parser()

    config = cli.parse_args(args)

    if config.name is None:
        import sys

        LOGGER.error("Experiment name must be specified")
        sys.exit(1)

    tune_ray(config)


if __name__ == "__main__":
    main()


def train(*args, **kwargs):
    print("training..")
    print(args)
    print(kwargs)
    print("done..")
