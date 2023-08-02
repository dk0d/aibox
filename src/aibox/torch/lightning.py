from functools import partial

from aibox.config import Config

try:
    import multiprocessing as mp
    from argparse import ArgumentParser

    import lightning.pytorch as L
    from lightning.pytorch.cli import LightningArgumentParser
    from rich import print
    from torch.utils.data import DataLoader, Dataset, IterableDataset

    from ..config import init_from_cfg, is_dict, is_list

except ImportError:
    print("lightning.pytorch required to import these utilities")
    exit(1)


def nondefault_trainer_args(opt):
    parser = ArgumentParser()
    parser = LightningArgumentParser.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# class SetupCallback(Callback):
#     def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
#         super().__init__()
#         self.resume = resume
#         self.now = now
#         self.logdir = logdir
#         self.ckptdir = ckptdir
#         self.cfgdir = cfgdir
#         self.config = config
#         self.lightning_config = lightning_config
#
#     def on_keyboard_interrupt(self, trainer, pl_module):
#         if trainer.global_rank == 0:
#             print("Summoning checkpoint.")
#             ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
#             trainer.save_checkpoint(ckpt_path)
#
#     def on_pretrain_routine_start(self, trainer, pl_module):
#         if trainer.global_rank == 0:
#             # Create logdirs and save configs
#             os.makedirs(self.logdir, exist_ok=True)
#             os.makedirs(self.ckptdir, exist_ok=True)
#             os.makedirs(self.cfgdir, exist_ok=True)
#
#             if "callbacks" in self.lightning_config:
#                 if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
#                     os.makedirs(
#                         os.path.join(self.ckptdir, "trainstep_checkpoints"),
#                         exist_ok=True,
#                     )
#             print("Project config")
#             print(OmegaConf.to_yaml(self.config))
#             OmegaConf.save(
#                 self.config,
#                 os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
#             )
#
#             print("Lightning config")
#             print(OmegaConf.to_yaml(self.lightning_config))
#             OmegaConf.save(
#                 OmegaConf.create({"lightning": self.lightning_config}),
#                 os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
#             )
#
#         else:
#             # ModelCheckpoint callback created log directory --- remove it
#             if not self.resume and os.path.exists(self.logdir):
#                 dst, name = os.path.split(self.logdir)
#                 dst = os.path.join(dst, "child_runs", name)
#                 os.makedirs(os.path.split(dst)[0], exist_ok=True)
#                 try:
#                     os.rename(self.logdir, dst)
#                 except FileNotFoundError:
#                     pass


# Adapted from LDM: https://github.com/CompVis/latent-diffusion/blob/main/main.py


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetFromConfig(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = init_from_cfg(config)

    def __repr__(self):
        return f"DatasetFromConfig({repr(self.dataset)})"

    def __str__(self):
        return f"DatasetFromConfig({str(self.dataset)})"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TransformFromConfig:
    def __repr__(self) -> str:
        return f"TransformFromConfig({repr(self.transforms)})"

    def __str__(self) -> str:
        return f"TransformFromConfig({str(self.transforms)})"

    def __init__(self, config):
        self.config = config
        if is_list(config):
            from torchvision.transforms import Compose

            self.transforms = Compose([init_from_cfg(t) for t in config])
        elif is_dict(config):
            self.transforms = init_from_cfg(config)

        else:
            raise NotImplementedError(f"Transforms must be a list or dict, got {type(config)}")

    def __call__(self, x):
        return self.transforms(x)


class DataModuleFromConfig(L.LightningDataModule):
    @property
    def train_dataset(self):
        return self.datasets["train"]

    @property
    def val_dataset(self):
        return self.datasets["val"]

    @property
    def test_dataset(self):
        return self.datasets["test"]

    @property
    def predict_dataset(self):
        return self.datasets["predict"]

    def __init__(
        self,
        batch_size,
        sample_shape,
        train: Config,
        validation: Config | None = None,
        test: Config | None = None,
        predict: Config | None = None,
        split_kwargs: Config | None = None,
        num_workers=None,
        # Transforms
        shuffle_test_loader=False,
        # use_worker_init_fn=False,
        shuffle_val_dataloader=False,
        wrap=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sample_shape = sample_shape
        self.dataset_configs = dict()
        self.split_kwargs = split_kwargs if split_kwargs is not None else dict()
        # self.use_worker_init_fn = use_worker_init_fn

        if num_workers is not None and num_workers == "batch_size":
            self.num_workers = batch_size * 2
        else:
            self.num_workers = num_workers if num_workers is not None else mp.cpu_count()

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            init_from_cfg(data_cfg, **self.split_kwargs)

    def setup(self, stage=None):
        self.datasets = {k: init_from_cfg(self.dataset_configs[k], **self.split_kwargs) for k in self.dataset_configs}
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _get_dataloader(self, split, shuffle=False):
        # is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        # init_fn = worker_init_fn
        # else:
        init_fn = None
        is_iterable_dataset = isinstance(self.datasets[split], IterableDataset)
        match split:
            case "test":
                shuffle = shuffle and (not is_iterable_dataset)
            case _:
                pass
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else shuffle,
            worker_init_fn=init_fn,
        )

    def _train_dataloader(self):
        return self._get_dataloader("train", True)

    def _val_dataloader(self, shuffle=False):
        return self._get_dataloader("validation", shuffle)

    def _test_dataloader(self, shuffle=False):
        return self._get_dataloader("test", shuffle)

    def _predict_dataloader(self, shuffle=False):
        return self._get_dataloader("predict", shuffle)


class LightningModuleFromConfig(L.LightningModule):
    def __init__(self, *, model, optimizers, schedulers, loss, **kwargs):
        """
        Assumes loss, optimizers, and schedulers are all configuration dictionaries
        """
        super().__init__()

        try:
            if loss is None:
                self.loss_fn = None
            else:
                self.loss_fn = init_from_cfg(loss)
        except Exception as e:
            print(f"[red bold]Error instantiating config: {loss}")
            print(f"Exception {e}")
            exit(0)

        self.model = init_from_cfg(model, **kwargs)
        self.optimizers_cfg = optimizers
        self.schedulers_cfg = schedulers
        self.current_device = None

    def configure_optimizers(self):
        optims = []
        scheds = []

        # see: https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#auto-wrapping
        # self.trainer.model.parameters(), # instead of self.model.parameters() for details -- for fully
        # sharded model training
        if is_list(self.optimizers_cfg):
            for cfg in self.optimizers_cfg:
                optims.append(init_from_cfg(cfg, self.parameters()))
        elif is_dict(self.optimizers_cfg):
            optims.append(init_from_cfg(self.optimizers_cfg, self.parameters()))
        else:
            raise NotImplementedError

        if self.schedulers_cfg is not None:
            if is_list(self.schedulers_cfg):
                assert len(optims) == len(self.schedulers_cfg), "scheduler list must be same length as optimizers"
                for i, cfg in enumerate(self.schedulers_cfg):
                    scheds.append(init_from_cfg(cfg, optims[i]))
            elif is_dict(self.schedulers_cfg):
                scheds.append(init_from_cfg(self.schedulers_cfg))
            else:
                raise NotImplementedError

        if len(scheds) > 0 and len(optims) > 0:
            return optims, scheds

        return optims[0]
