import os
import sys

import cv2
import lightning as L
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from carformer.data import get_datasets
from carformer.ponderer import Ponderer
from carformer.utils import (
    WeightedDistributedSampler,
)
from carformer.utils.distributed import get_rank
from carformer.visualization.visutils import (
    visualize_input_from_batch,
    visualize_trajectory_action_predictions,
)


class PondererLit(L.LightningModule):
    def __init__(self, ponderer):
        super(PondererLit, self).__init__()
        self.ponderer = ponderer

        self.params = ponderer.cfg

        self.save_hyperparameters(ponderer.cfg.to_dict())

        print("Save dir: ", self.params.save_dir)

    def setup(self, stage):
        if stage != "fit":
            return

        folder = self.params.save_dir

        if self.trainer.is_global_zero:
            os.makedirs(folder, exist_ok=True)
            os.makedirs(os.path.join(folder, "train_predictions"), exist_ok=True)
            os.makedirs(os.path.join(folder, "val_predictions"), exist_ok=True)

            # Save the model config and training args
            self.params.save_pretrained(folder)
            # Save all training arguments
            env_vars = []

            vars_to_save = [
                "HF_ENDPOINT",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]

            for var in vars_to_save:
                if var in os.environ:
                    env_vars.append(f"{var}={os.environ[var]}")

            # Detect if ddp or not
            if "WORLD_SIZE" in os.environ:
                run_command = f"torchrun --nproc_per_node={os.environ['WORLD_SIZE']}"
            else:
                run_command = "python"

            # Save the command used to run the training into the file {folder}/command.sh
            with open(os.path.join(folder, "command.sh"), "w") as f:
                all_args = env_vars + [run_command] + sys.argv
                f.write(" ".join(all_args))
                f.write("\n")
                # If wandb is used, log the url to the run
                try:
                    url = wandb.run.get_url()
                    f.write(f"Wandb URL: {url}")
                    f.write("\n")
                except:
                    pass

    def training_step(self, batch, batch_idx):
        preds, loss, pred_labels, preprocessed_inputs = self.ponderer(
            batch, return_labels=True, return_inputs=True
        )

        loss = {k: v.mean() for k, v in loss.items()}

        self.log_dict(
            {f"train/loss/{k}": v for k, v in loss.items()},
            sync_dist=True,
            prog_bar=True,
        )
        rank = get_rank()

        world_size = self.trainer.world_size
        # import ipdb; ipdb.set_trace()

        if (
            batch_idx < 16
            and self.params.visualize
            and self.trainer.current_epoch > self.params.visualize_start_epoch
            and (self.trainer.current_epoch % self.params.visualize_interval == 0)
            and rank < 4  # only log from first 4 ranks
        ):
            for idd in range(max(self.params.hyperparams.batch_size // 8, 1)):
                # import ipdb; ipdb.set_trace()

                ac_val = batch["action"][0][0, 20][0].item()

                impath = visualize_input_from_batch(
                    batch,
                    idd,
                    preds,
                    pred_labels,
                    self.params.save_dir,
                    f"ep{self.trainer.current_epoch}_st{rank*8 + idd + world_size*4*batch_idx}_ac_{ac_val:.2f}",
                    self.ponderer,
                    "train",
                )
                dir_preds = os.path.join(self.params.save_dir, "train_predictions")

                os.makedirs(dir_preds, exist_ok=True)

                path2 = visualize_trajectory_action_predictions(
                    batch,
                    preds,
                    labels=pred_labels,
                    save_dir=dir_preds,
                    model=self.ponderer,
                    save_suffix="ep{}_st{}".format(
                        self.trainer.current_epoch,
                        rank * 8 + idd + world_size * 4 * batch_idx,
                    ),
                    save_idx=idd,
                    action_source="transformer-regression",
                    visualize_gt=True,
                )
                if not preds["bev"] is None:
                    pred_heatmaps = preds["bev"].float().detach().cpu().numpy()[idd]

                    label_heatmaps = (
                        pred_labels["bev_mask"].float().detach().cpu().numpy()[idd]
                    )

                    size_per_patch = label_heatmaps.shape[-1] // 2

                    patch_1 = (
                        label_heatmaps[:size_per_patch].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255
                    patch_2 = (
                        label_heatmaps[size_per_patch:].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255

                    patch_1_pred = (
                        pred_heatmaps[:size_per_patch].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255
                    patch_2_pred = (
                        pred_heatmaps[size_per_patch:].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255

                    patches = np.concatenate([patch_1, patch_2], axis=1)
                    patches_pred = np.concatenate([patch_1_pred, patch_2_pred], axis=1)

                    # Make pred red, label green
                    patch = np.stack(
                        [patches, patches_pred, np.zeros_like(patches)], axis=-1
                    )

                    cv2.imwrite(
                        os.path.join(
                            dir_preds,
                            f"ep{self.trainer.current_epoch}_st{rank*8 + idd + world_size*4*batch_idx}_patch.png",
                        ),
                        patch,
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, pred_labels, preprocessed_inputs = self.ponderer(
            batch, return_labels=True, return_inputs=True
        )

        loss = {k: v.mean() for k, v in loss.items()}

        self.log_dict({f"val/loss/{k}": v for k, v in loss.items()}, sync_dist=True)
        rank = get_rank()

        world_size = self.trainer.world_size

        if (
            batch_idx < 16
            and self.params.visualize
            and self.trainer.current_epoch > self.params.visualize_start_epoch
            and (self.trainer.current_epoch % self.params.visualize_interval == 0)
            and rank < 4  # only log from first 4 ranks
        ):
            for idd in range(max(self.params.hyperparams.batch_size // 8, 1)):
                impath = visualize_input_from_batch(
                    batch,
                    idd,
                    preds,
                    pred_labels,
                    self.params.save_dir,
                    f"ep{self.trainer.current_epoch}_st{rank*8 + idd + world_size*4*batch_idx}",
                    self.ponderer,
                    "val",
                )
                dir_preds = os.path.join(self.params.save_dir, "val_predictions")

                os.makedirs(dir_preds, exist_ok=True)

                path2 = visualize_trajectory_action_predictions(
                    batch,
                    preds,
                    labels=pred_labels,
                    save_dir=dir_preds,
                    model=self.ponderer,
                    save_suffix="ep{}_st{}".format(
                        self.trainer.current_epoch,
                        rank * 8 + idd + world_size * 4 * batch_idx,
                    ),
                    save_idx=idd,
                    action_source="transformer-regression",
                )
                if not preds["bev"] is None:
                    pred_heatmaps = preds["bev"].float().detach().cpu().numpy()[idd]

                    label_heatmaps = (
                        pred_labels["bev_mask"].float().detach().cpu().numpy()[idd]
                    )

                    size_per_patch = label_heatmaps.shape[-1] // 2

                    patch_1 = (
                        label_heatmaps[:size_per_patch].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255
                    patch_2 = (
                        label_heatmaps[size_per_patch:].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255

                    patch_1_pred = (
                        pred_heatmaps[:size_per_patch].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255
                    patch_2_pred = (
                        pred_heatmaps[size_per_patch:].reshape(
                            int(size_per_patch**0.5), int(size_per_patch**0.5)
                        )
                        > 0.5
                    ) * 255

                    patches = np.concatenate([patch_1, patch_2], axis=1)
                    patches_pred = np.concatenate([patch_1_pred, patch_2_pred], axis=1)

                    # Make pred red, label green
                    patch = np.stack(
                        [patches, patches_pred, np.zeros_like(patches)], axis=-1
                    )

                    cv2.imwrite(
                        os.path.join(
                            dir_preds,
                            f"ep{self.trainer.current_epoch}_st{rank*8 + idd + world_size*4*batch_idx}_patch.png",
                        ),
                        patch,
                    )

        return loss

    def configure_optimizers(self):
        return_dict = {}

        params = self.parameters()

        opt_kwargs = self.params.hyperparams.optimizer.kwargs

        if "weight_decay" in opt_kwargs:
            # Pop weight decay from optimizer kwargs
            weight_decay = opt_kwargs.pop("weight_decay")

            # Create optimizer groups
            optim_groups = self.create_optimizer_groups(weight_decay)
        else:
            optim_groups = params

        opt = getattr(torch.optim, self.params.hyperparams.optimizer.name)(
            optim_groups,
            lr=self.params.hyperparams.lr,
            **self.params.hyperparams.optimizer.kwargs,
        )
        stepping_batches = self.trainer.estimated_stepping_batches

        from torch.optim.lr_scheduler import (
            CosineAnnealingWarmRestarts,
        )

        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=stepping_batches // 30, T_mult=2, eta_min=1e-6
        )

        return_dict["optimizer"] = opt

        return_dict["lr_scheduler"] = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return return_dict

    def train_dataloader(self):
        train_dataset, _ = get_datasets(self.params, self.ponderer, splits=["train"])
        subsample_ratio = self.params.dataset.subsample_ratio
        if self.params.training.weighted_sampling:
            initial_weights = train_dataset.getweights()

            self.sample_weights = initial_weights

            bucket_names = train_dataset.get_bucket_names()
            self.bucket_names = bucket_names

            bucket_config = self.params.training.bucket_weights

            if bucket_config.type == "uniform":
                weights = np.asarray(initial_weights)
                weights = (weights / weights.sum(0)).mean(-1)
                subsample_ratio *= bucket_config.total_ratio
            if bucket_config.type == "preferturns":
                weights = np.asarray(initial_weights)
                weights = (
                    (weights / weights.sum(0)) * np.asarray(bucket_config.weights)
                ).sum(-1) / np.asarray(bucket_config.weights).sum()
                subsample_ratio *= bucket_config.total_ratio
            else:
                raise NotImplementedError(
                    f"Bucketing type {bucket_config.type} not implemented yet"
                )
        else:
            weights = None

        sampler = WeightedDistributedSampler(
            train_dataset,
            subsample_ratio,
            shuffle=True if not self.params.overfit else False,
            weights=weights,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.hyperparams.batch_size,
            sampler=sampler,
            num_workers=self.params.num_workers,
        )

        self.train_loader = train_loader

        return train_loader

    def val_dataloader(self):
        _, val_dataset = get_datasets(self.params, self.ponderer, splits=["val"])

        val_sampler = WeightedDistributedSampler(
            val_dataset,
            shuffle=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params.hyperparams.batch_size,
            sampler=val_sampler,
            num_workers=self.params.num_workers,
        )

        return val_loader

    @staticmethod
    def from_pretrained(path, epoch=30, deepspeed="auto", apply_deepspeed=True):
        assert os.path.exists(path), "Path {} does not exist".format(path)

        if deepspeed == "auto":
            # Infer whether the checkpoint is in deepspeed format
            if os.path.exists(os.path.join(path, "zero_to_fp32.py")):
                deepspeed = True
            else:
                deepspeed = False

        config_path = os.path.join(path, "config.json")

        assert os.path.exists(config_path), "Config path {} does not exist".format(
            config_path
        )

        config = CarformerConfig.from_pretrained(config_path)

        # Fix quantizer paths if needed
        import inspect

        import carformer

        carformer_path = os.path.dirname(inspect.getfile(carformer))
        # Go one layer up
        carformer_path = os.path.dirname(carformer_path)

        # Replace encoder path
        if "rgb_backbone" in config.training:
            if config.training.rgb_backbone.frozen:
                config.training.rgb_backbone.model_path = os.path.join(
                    carformer_path, config.training.rgb_backbone.model_path_rel
                )
            else:
                # Do not load path, the weights and config should already exist.
                config.training.rgb_backbone.model_path = None

        model = Ponderer(config)

        if epoch in ["last", "best"]:
            checkpoint_path = os.path.join(path, "{}_model.pt".format(epoch))
        else:
            if epoch is not None:
                if deepspeed:
                    checkpoint_path = os.path.join(
                        path, "{}".format(epoch)
                    )  # Deepspeed style
                else:
                    checkpoint_path = os.path.join(
                        path, "epochs", "epoch_{}.pt".format(epoch)
                    )
            else:
                checkpoint_path = None

        if deepspeed == True:
            import deepspeed as ds

            arg_dict = {}
            if checkpoint_path is not None:
                arg_dict["checkpoint"] = checkpoint_path

            print("Loading args ", arg_dict)
            model = ds.init_inference(model, arg_dict)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            dtype = next(iter(checkpoint["model"].values())).dtype
            model.to(dtype)
            model.load_state_dict(checkpoint["model"], strict=True)
            if apply_deepspeed:
                import deepspeed as ds

        return model

    def on_train_epoch_start(self):
        # Only do this on rank 0
        if not self.trainer.is_global_zero:
            return

        if self.params.training.weighted_sampling:
            weights = self.sample_weights
            self.train_loader.sampler.__iter__()

            indices = self.train_loader.sampler.last_indices

            bucket_names = self.bucket_names

            dist = weights[indices].sum(0) / (
                self.train_loader.sampler.num_samples
                * self.train_loader.sampler.num_replicas
            )
            baseline_dist = weights.sum(0) / len(weights)

            print("Weighted sampling statistics, epoch: ", self.trainer.current_epoch)
            for bucket_name, dist_val, baseline_val in zip(
                bucket_names, dist, baseline_dist
            ):
                print(
                    f"Bucket {bucket_name}: {dist_val*100:.2f}% (Baseline: {baseline_val*100:.2f}%)",
                    end="\t",
                )

    def create_optimizer_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is
        being very defensive:
        We are separating out all parameters of the model into two buckets:
        those that will experience
        weight decay for regularization and those that won't
        (biases, and layernorm/embedding weights).
        We are then returning the optimizer groups.
        """

        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.BatchNorm2d,
        )
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # if "attn_pool" in fpn:
                #     import ipdb; ipdb.set_trace()
                # print(fpn)
                # if len(no_decay&decay) > 0:
                #     import ipdb; ipdb.set_trace()
                # fpn = pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif "attn_pool" in fpn or "vqvae" in fpn:
                    no_decay.add(fpn)
                elif "norm.weight" in fpn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif (
                    pn.endswith("weight") and "conv." in pn
                ):  # Add decay for convolutional layers.
                    decay.add(fpn)
                elif pn.endswith("weight") and ".bn" in pn:  # No decay for batch norms.
                    no_decay.add(fpn)
                elif pn.endswith("weight") and ".ln" in pn:  # No decay for layer norms.
                    no_decay.add(fpn)
                elif (
                    pn.endswith("weight") and "downsample.0.weight" in pn
                ):  # Conv2D layer with stride 2
                    decay.add(fpn)
                elif pn.endswith("weight") and "downsample.1.weight" in pn:  # BN layer
                    no_decay.add(fpn)
                elif pn.endswith("weight") and ".attn" in pn:  # Attention linear layers
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") and "channel_to_" in pn
                ):  # Convolutional layers for channel change
                    decay.add(fpn)
                elif pn.endswith("weight") and ".mlp" in pn:  # MLP linear layers
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") and "target_speed_network" in pn
                ):  # MLP linear layers
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") and "join." in pn and not ".norm" in pn
                ):  # MLP layers
                    decay.add(fpn)
                elif (
                    pn.endswith("weight") and "join." in pn and ".norm" in pn
                ):  # Norm layers
                    no_decay.add(fpn)
                elif pn.endswith("weight") and "layernorm" in pn:  # Norm layers
                    no_decay.add(fpn)
                elif pn.endswith("weight") and ".norm" in fpn:  # Norm layers
                    no_decay.add(fpn)
                elif "class_embedding" in fpn:  # cls embeds
                    no_decay.add(fpn)
                elif pn.endswith("_ih") or pn.endswith("_hh"):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("_emb") or "_token" in pn:
                    no_decay.add(fpn)
                elif pn.endswith("_embed"):
                    no_decay.add(fpn)
                elif "pos_embed" in pn:
                    no_decay.add(fpn)
                elif "patch_embed" in pn:
                    decay.add(fpn)
                elif "bias_ih_l0" in pn or "bias_hh_l0" in pn:
                    no_decay.add(fpn)
                elif "weight_ih_l0" in pn or "weight_hh_l0" in pn:
                    decay.add(fpn)
                elif "_query" in pn or "weight_hh_l0" in pn:
                    no_decay.add(fpn)
                elif "proj_weight" in pn:
                    decay.add(fpn)
                elif "valid_bev_pixels" in pn:
                    no_decay.add(fpn)
                elif "ls1" in pn or "ls2" in pn:
                    no_decay.add(fpn)
                elif "position_embedding" in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {str(param_dict.keys() - union_params)} were not "
            f"separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def on_train_batch_end(self, *args, **kwargs):

        ema_enabled = self.ponderer.bev_encoder.ema_enabled
        disable_ema_updates = self.ponderer.bev_encoder.disable_ema_update
        if disable_ema_updates:
            return

        if ema_enabled:
            ema_every_steps = getattr(
                self.ponderer.config.training, "ema_every_steps", 1
            )
            ema_start = getattr(self.ponderer.config.training, "ema_start", 0)
            ema_end_epoch = getattr(self.ponderer.config.training, "ema_end_epoch", -1)

            if self.global_step % ema_every_steps == 0 and self.global_step > ema_start:
                # Stop if end epoch is reached
                if ema_end_epoch == -1 or self.trainer.current_epoch < ema_end_epoch:
                    ema_decay = self.ponderer.config.training.ema_decay

                    vision_encoder = self.ponderer.bev_encoder

                    model = vision_encoder.model

                    ema_model = vision_encoder.ema_model
                    self.ponderer.ema_update(model, ema_model, beta=ema_decay)
