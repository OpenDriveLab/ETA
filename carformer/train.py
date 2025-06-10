from carformer.ponderer import Ponderer
from carformer.ponderer_lit import PondererLit

from carformer.utils import seed_everything
import os
from carformer.config import config_init, CarformerConfig
from omegaconf import OmegaConf
import hydra
from skit.distributed.deepspeed_utils import ZeroLightModelCheckpoint

config_init()


@hydra.main(version_base="1.1", config_path="carformer/config", config_name="config")
def main(cfg):
    seed_everything(cfg.seed)

    args = cfg

    if args.ckpt_path is not None:
        print("Loading model from checkpoint")
        cfg.save_dir = os.path.dirname(args.ckpt_path)

    config = CarformerConfig.from_hydra(args)

    print(config)

    model = Ponderer(config)

    model = PondererLit(model).train()

    ds_cfg = OmegaConf.to_container(cfg.deepspeed, resolve=True)

    folder = cfg.save_dir

    from lightning import Trainer
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        LearningRateMonitor,
    )
    from lightning.pytorch.strategies import DeepSpeedStrategy

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if (not args.overfit) or args.force_save:
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=os.path.join(folder),
                    filename="last_model",
                    every_n_epochs=0,
                    save_last=True,
                ),
            ]
        )

        callbacks.extend(
            [
                ZeroLightModelCheckpoint(
                    dirpath=os.path.join(folder, "epochs"),
                    filename="{epoch}",
                    monitor="val/loss/wp_loss",
                    mode="min",
                    every_n_epochs=args.save_every,
                    save_top_k=-1,
                    save_weights_only=True,
                    start_save_epoch=args.start_saving_epoch,
                )
            ]
        )

    if args.wandb_tag:
        tags = [args.wandb_tag]
    else:
        tags = []

    if not (args.overfit or args.debug) or args.force_log:
        log = True
    else:
        log = False

    trainer = Trainer(
        accelerator="gpu" if not args.cpu else "cpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy=(DeepSpeedStrategy(config=ds_cfg)),
        max_epochs=args.hyperparams.num_epochs,
        logger=(
            WandbLogger(
                project=args.logging.project,
                entity=args.logging.entity,
                mode=args.logging.mode,
                tags=tags,
                offline=True,
                save_code=True,
            )
            if log
            else None
        ),
        callbacks=callbacks,
        accumulate_grad_batches=args.hyperparams.gradient_accumulation_steps,
        use_distributed_sampler=False,
        overfit_batches=args.overfit_batches if args.overfit else 0.0,
        limit_val_batches=0 if args.overfit else 1.0,
        log_every_n_steps=1 if args.overfit else 25,
        enable_checkpointing=False if args.overfit else True,
    )

    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
