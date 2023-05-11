try:
    import pytorch_lightning as pl
    from pytorch_lightning.cli import LightningArgumentParser
    import omegaconf as oc
    from rich import print

    from argparse import ArgumentParser

    from ..config import init_from_cfg
    
except ImportError:
    print("pytorch_lightning required to import these utilities")
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

class AIBoxLightningModule(pl.LightningModule):
    
    def __str__(self) -> str:
        return f"{self.model.__class__.__name__}"

    def __repr__(self):
        return super().__str__()

    def __init__(self, optimizers, schedulers, loss, **kwargs):
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


        self.optimizers_cfg = optimizers
        self.schedulers_cfg = schedulers
        self.current_device = None

    def configure_optimizers(self):

        optims = []
        scheds = []

        # see: https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#auto-wrapping
        # self.trainer.model.parameters(), # instead of self.model.parameters() for details -- for fully
        # sharded model training
        if oc.OmegaConf.is_list(self.optimizers_cfg):
            for cfg in self.optimizers_cfg:
                optims.append(init_from_cfg(cfg, self.parameters()))
        elif oc.OmegaConf.is_dict(self.optimizers_cfg):
            optims.append(init_from_cfg(self.optimizers_cfg, self.parameters()))
        else:
            raise NotImplementedError

        if self.schedulers_cfg is not None:
            if oc.OmegaConf.is_list(self.schedulers_cfg):
                assert len(optims) == len(self.schedulers_cfg), "scheduler list must be same length as optimizers"
                for i, cfg in enumerate(self.schedulers_cfg):
                    scheds.append(init_from_cfg(cfg, optims[i]))
            elif oc.OmegaConf.is_dict(self.schedulers_cfg):
                scheds.append(init_from_cfg(self.schedulers_cfg))
            else:
                raise NotImplementedError

        if len(scheds) > 0 and len(optims) > 0:
            return optims, scheds


        return optims[0]

    # def _step(self, batch, batchIdx, optimizerIdx=0):
    #     x, prior, params = resolveBatch(batch)
    #     self.current_device = x.device
    #
    #     if isinstance(self.loss_fn, StatefulLoss):
    #         # some forward passes use values from loss object
    #         # see Categorical VAE & Loss as example
    #         params.update(**self.loss_fn.lossState())
    #
    #     out = self.model(x, prior, **params)
    #     loss = self.loss_fn(
    #         *out,
    #         kld_weight=self.kld_weight,
    #         batchIdx=batchIdx,
    #         optimizerIdx=optimizerIdx,
    #     )
    #     return loss

    # def _prefix_log(self, prefix, loss: dict):
    #     self.log_dict({f"{prefix}/{key}": val.item() for key, val in loss.items()}, sync_dist=True)


    # def training_step(self, batch, batchIdx, optimizerIdx=0):
    #     loss = self._step(batch, batchIdx, optimizerIdx)
    #     self._prefix_log("train", loss)
    #     return loss["loss"]

    # def validation_step(self, batch, batchIdx, optimizerIdx=0):
    #     loss = self._step(batch, batchIdx, optimizerIdx)
    #     self._prefix_log("val", loss)
    #     return loss["loss"]

    # def on_validation_end(self):
    #     self.sampleImages()

    # def test_step(self, batch, batchIdx, optimizerIdx=0):
    #     loss = self._step(batch, batchIdx, optimizerIdx)
    #     self._prefix_log("test", loss)
    #     return loss["loss"]
