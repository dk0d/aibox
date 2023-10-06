from typing import Any

import numpy as np
from omegaconf import OmegaConf

from aibox.logger import get_logger

# import os
# from aibox.slurm.hpc import HyperOptWrapper, SlurmCluster

# from lightning import seed_everything
# from lightning.strategies.ddp import DDPStrategy
# from ray.tune.integration.lightning import TuneReportCallback
from aibox.torch.training import build_cli_parser, train_and_test
from aibox.utils import is_list, print

# os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

# from ax.plot.contour import plot_contour
# from ax.plot.trace import optimization_trace_single_method
# from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render, init_notebook_plotting
# from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN


LOGGER = get_logger(__name__)


def tune_train(tune_config, config):
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    tune_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in tune_config.items()])
    train_config = OmegaConf.merge(config, tune_config)
    results = train_and_test(train_config)
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
        values = list(OmegaConf.to_container(values))

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
                return tune.uniform(*values)
            return {"bounds": values, "type": "range", "value_type": "float"}
        case "loguniform":
            if mode == "ray":
                assert tune is not None
                return tune.loguniform(*values)
            return {"bounds": values, "type": "range", "log_scale": True, "value_type": "float"}
        case "randint":
            if mode == "ray":
                assert tune is not None
                return tune.randint(*values)
            return {"bounds": values, "type": "range", "value_type": "int"}
        case "randn":
            if mode == "ray":
                assert tune is not None
                return tune.randn(*values)
            return {"bounds": values, "type": "range", "value_type": "float"}
        case "quniform":
            if mode == "ray":
                assert tune is not None
                return tune.quniform(*values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qloguniform":
            if mode == "ray":
                assert tune is not None
                return tune.qloguniform(*values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qrandint":
            if mode == "ray":
                assert tune is not None
                return tune.qrandint(*values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "qrandn":
            if mode == "ray":
                assert tune is not None
                return tune.qrandn(*values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "grid_search":
            if mode == "ray":
                assert tune is not None
                return tune.grid_search(values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
        case "rand_grid_search":
            if mode == "ray":
                assert tune is not None
                return tune.rand_grid_search(values)
            raise NotImplementedError(f"{_type} not implemented for Ax")
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


def tune_ray(config):
    """
    Initialize Ray Tune and run hyperparameter tuning.

    Warning: RayTune support for hyperparameter tuning not fully tested yet
        Tested only on single, local machine
        See more at [MLFlow Ray Example](https://docs.ray.io/en/latest/tune/examples/includes/mlflow_ptl_example.html)

    Attributes:
        config: A dictionary containing the configuration for the experiment.

    """
    from ray import air, tune

    # from ray.tune.search.ax import AxSearch
    from ray.air.integrations.mlflow import MLflowLoggerCallback

    tune_config = gather_search_spaces_ray(config)
    trainable = tune.with_parameters(tune_train, config=config)
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=config.tuner.num_samples,
        ),
        run_config=air.RunConfig(
            name=config.name,
            storage_path="~/ray_results",
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name=config.name,
                    tracking_uri=config.logger.tracking_uri,
                ),
            ],
        ),
        param_space=tune_config,
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
#         exit(1)
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
        LOGGER.error("Experiment name must be specified")
        exit(1)

    tune_ray(config)


if __name__ == "__main__":
    main()


def train(*args, **kwargs):
    print("training..")
    print(args)
    print(kwargs)
    print("done..")
