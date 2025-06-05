import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoImageProcessor, AutoConfig, CONFIG_MAPPING

try:
    from transformers.models.auto.image_processing_auto import (
        image_processor_class_from_name,
    )
except ImportError:
    from transformers.models.auto.image_processing_auto import (
        get_image_processor_class_from_name as image_processor_class_from_name,
    )

rgb_models = {}


class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        # Load config
        if isinstance(config, str):
            with open(config, "r") as file:
                try:
                    config = yaml.safe_load(file)
                    self.config = config
                except yaml.YAMLError as exc:
                    print(exc)
                    raise exc
        else:
            self.config = config

        pretrained_path = self.config.get("model_path", None)
        pretrained_override_kwargs = self.config.get("override_kwargs", {})

        self.ema_enabled = self.config.get("ema_enabled", False)
        self.keep_time_dim = self.config.get("keep_time_dim", False)

        if pretrained_path is not None:
            print(
                "Overriding main model config using kwargs: ",
                pretrained_override_kwargs,
            )
            self.cfg = AutoConfig.from_pretrained(
                pretrained_path, **pretrained_override_kwargs
            )

            self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
            self.model = AutoModel.from_pretrained(
                pretrained_path, **pretrained_override_kwargs
            )

            self.config["override_kwargs"] = {}
            self.config["model_config"] = self.cfg.to_dict()
            self.config["processor_config"] = self.processor.to_dict()

            if self.ema_enabled:
                self.ema_model = AutoModel.from_pretrained(
                    pretrained_path, **pretrained_override_kwargs
                )
            else:
                self.ema_model = None
        else:
            assert (
                "model_config" in self.config
            ), "model_config must be provided if model_path is not provided"
            assert (
                "processor_config" in self.config
            ), "processor_config must be provided if model_path is not provided"

            model_type = self.config["model_config"]["model_type"]
            image_processor_type = self.config["processor_config"][
                "image_processor_type"
            ]

            self.cfg = CONFIG_MAPPING[model_type].from_dict(self.config["model_config"])
            self.processor = image_processor_class_from_name(
                image_processor_type
            ).from_dict(self.config["processor_config"])
            self.model = AutoModel.from_config(self.cfg)

            if self.ema_enabled:
                self.ema_model = AutoModel.from_config(self.cfg)
            else:
                self.ema_model = None

        self.projector = nn.Linear(
            self.cfg.hidden_size, self.config.projection_dim
        )  # self.cfg: AutoConfig, backbone specific. self.config: user provided config, contains main backbone hidden dim

        self.select_layer = self.config.get("select_layer", None)
        # Dummy rgb encoder utils
        self.dummy_output = self.config.get("dummy_output", False)

        self.move_channels_last_and_flatten = self.config.get(
            "move_channels_last_and_flatten", False
        )
        self.ignore_cls = self.config.get("ignore_cls", True)

        if self.select_layer is None:
            self.select_layer = -2
            print("Warning: select_layer not provided, using default -2")

        init_config = self.config.get("init_from_ckpt", None)
        if init_config is not None and init_config["enabled"]:
            init_config = init_config.copy()
            self.config["init_from_ckpt"]["enabled"] = False  # Do not init again
            ckpt = torch.load(init_config["ckpt_path"], map_location="cpu")
            if init_config["model"]:
                mdl_dct = {
                    k[len("model.") :]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.")
                }

                self.model.load_state_dict(mdl_dct, strict=True)

            if init_config["ema_model"] and self.ema_enabled:
                ema_dct = {
                    k[len("model.") :]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.")
                }
                self.ema_model.load_state_dict(ema_dct, strict=True)

            if init_config["projector"]:
                proj_dct = {
                    k[len("projector.") :]: v
                    for k, v in ckpt.items()
                    if k.startswith("projector.")
                }
                self.projector.load_state_dict(proj_dct, strict=True)

            self.config["frozen"]["projector"] = init_config["projector"]
            self.config["frozen"]["model"] = init_config["model"]
            self.config["frozen"]["ema_model"] = init_config["ema_model"]

        self.try_to_truncate_layers = self.config.get("try_to_truncate_layers", False)

        if self.try_to_truncate_layers:
            import transformers

            if isinstance(
                self.model, transformers.models.clip.modeling_clip.CLIPVisionModel
            ):
                layer_to_select = self.select_layer
                if layer_to_select < 0:
                    layer_to_select = (
                        len(self.model.vision_model.encoder.layers) + layer_to_select
                    )

                self.model.vision_model.encoder.layers = (
                    self.model.vision_model.encoder.layers[: layer_to_select + 1]
                )

                if self.ema_enabled:
                    self.ema_model.vision_model.encoder.layers = (
                        self.ema_model.vision_model.encoder.layers[
                            : layer_to_select + 1
                        ]
                    )

                self.select_layer = -1
            else:
                print(
                    "Warning: try_to_truncate_layers is set to True, but model is not CLIP. "
                    "This will not truncate any layers."
                )

        frozen = self.config.get("frozen", False)
        self.disable_ema_update = False
        if isinstance(frozen, bool):
            if frozen:
                self.model.requires_grad_(False)
                self.model.eval()
                self.frozen = True
            else:
                self.frozen = False
            self.frozen_dict = {}
        else:
            if frozen["model"]:
                self.model.requires_grad_(False)
                self.model.eval()
            if frozen["ema_model"]:
                if not self.ema_enabled:
                    self.model.requires_grad_(False)
                    self.model.eval()
                else:
                    self.ema_model.requires_grad_(False)
                    self.ema_model.eval()
                    self.disable_ema_update = True
            if frozen["projector"]:
                self.projector.requires_grad_(False)
            self.frozen = False
            self.frozen_dict = {
                "model": frozen["model"]
                or (frozen["ema_model"] and not self.ema_enabled),
                "ema_model": frozen["ema_model"],
                "projector": frozen["projector"],
            }

        self.output_whole = self.config["outputs"]["whole"]
        self.output_patches = self.config["outputs"]["patches"]

        # center crop of width self.config.input_size*2, height self.config.input_size
        def center_crop(x):
            wdesired = self.config.input_size * 2
            hdesired = self.config.input_size
            wcur = x.shape[-2]
            hcur = x.shape[-3]

            return x[
                ...,
                (hcur - hdesired) // 2 : (hcur + hdesired) // 2,
                (wcur - wdesired) // 2 : (wcur + wdesired) // 2,
                :,
            ]

        self.center_crop = center_crop

        self.camera_embeddings = nn.Embedding(2, self.cfg.hidden_size)

        self.masking_rate = self.config.get("masking_rate", 0.0)

        self.do_downsample = self.config.get("downsample", False)
        self.downsample_type = self.config.get("downsample_type", "conv")

        if self.do_downsample:
            if self.downsample_type == "conv":
                self.downsample = nn.Conv2d(
                    self.cfg.hidden_size,
                    self.cfg.hidden_size,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
            elif self.downsample_type == "avgpool":
                self.downsample = nn.AvgPool2d(2)
            else:
                raise ValueError("Invalid downsample type")
        else:
            self.downsample = None

        # Display warning that output_whole, patches is ignored currently and will be implemented later
        print(
            "Warning: output_whole, output_patches is ignored currently and will be implemented later"
        )

    def forward(self, x, y=None, ema=False):
        # Dims: B, T, NPatch, 3, H, W
        if self.frozen:
            self.model.eval()

            with torch.no_grad():
                return self.encode(x, y, ema=ema)
        else:
            return self.encode(x, y, ema=ema)

    def encode(self, x, y=None, ema=False):
        # print("x shape: ", x.shape, "y shape: ", y.shape if y is not None else None)
        # x = self.center_crop(x)  # TODO: This is too hacky, fix this
        original_shape = x.shape
        if len(original_shape) == 5:
            T = original_shape[1]
            x = x.reshape(
                original_shape[0] * original_shape[1], *original_shape[2:]
            )  # B*T, H, W, 3
        else:
            T = 1
        # Encode
        prepped_x = self.processor(
            list(x.squeeze(1).permute(0, 3, 1, 2).half()), return_tensors="pt"
        )["pixel_values"]
        prepped_y = (
            self.processor(
                list(y.squeeze(1).permute(0, 3, 1, 2).half()),
                return_tensors="pt",
                do_normalize=False,
                do_convert_rgb=False,
            )["pixel_values"]
            if y is not None
            else None
        )

        # Prep the inputs before feeding to the vision encoders
        # Remove first patch, which is the entire image
        prepped_x = prepped_x[:, 1:]
        if self.frozen_dict.get("ema_model", False) and self.ema_enabled:
            self.ema_model.eval()

        if self.frozen_dict.get("model", False):
            self.model.eval()

        if prepped_y is not None:
            prepped_y = prepped_y[:, 1:]

        B, Npatch = prepped_x.shape[:2]
        By = prepped_y.shape[0] if prepped_y is not None else None

        if self.dummy_output:
            return self.projector(
                torch.zeros(B, Npatch, self.cfg.hidden_size)
                .to(self.model.device)
                .to(self.model.dtype)
            )

        # Flatten, encode
        if self.keep_time_dim:
            prepped_x = prepped_x.reshape(B // T, T, *prepped_x.shape[1:])
            # swap dim 1 and 2
            prepped_x = prepped_x.permute(0, 2, 1, 3, 4, 5)

            if prepped_y is not None:
                prepped_y = prepped_y.reshape(By, 1, *prepped_y.shape[1:])
                prepped_y = prepped_y.permute(0, 2, 1, 3, 4, 5)

        elif ema and self.ema_enabled:
            vision_feats = self.ema_model(
                prepped_x.flatten(0, 1)
                .to(self.ema_model.device)
                .to(self.ema_model.dtype),
                output_hidden_states=True,
            )["hidden_states"]
        else:

            vision_feats = self.model(
                prepped_x.flatten(0, 1).to(self.model.device).to(self.model.dtype),
                output_hidden_states=True,
            )["hidden_states"]

        if prepped_y is not None:
            with torch.no_grad():
                if hasattr(self.model, "vision_model"):
                    vision_patch_labels = (
                        self.model.vision_model.embeddings.patch_embedding(
                            prepped_y.flatten(0, 1)
                            .to(self.model.device)
                            .to(self.model.dtype)
                        )
                    )
                else:
                    raise ValueError(
                        "Model does not have a patcher, cannot process labels"
                    )

                if self.do_downsample:
                    vision_patch_labels = self.downsample(vision_patch_labels)

                vision_patch_labels = vision_patch_labels.abs().sum(1).flatten(1, 2)

        if self.ignore_cls:
            # Vision feat to use, default from llama, ignore cls
            vision_feats = vision_feats[self.select_layer][:, 1:]
        else:
            assert y is None, "y is not None, but ignore_cls is False"
            assert self.do_downsample is False, "Cannot downsample with cls token"
            vision_feats = vision_feats[self.select_layer]

        if self.move_channels_last_and_flatten:
            vision_feats = vision_feats.permute(0, 2, 3, 1).flatten(1, 2)

        if self.do_downsample:
            vision_feat_side_size = int(vision_feats.shape[1] ** 0.5)

            vision_feats = vision_feats.view(
                B * Npatch, vision_feat_side_size, vision_feat_side_size, -1
            )
            # Permute hidden dim to channel dim
            vision_feats = vision_feats.permute(0, 3, 1, 2)

            vision_feats = self.downsample(vision_feats)

            vision_feats = vision_feats.permute(0, 2, 3, 1).flatten(
                1, 2
            )  # B*Npatch, H, W, C -> B*Npatch, H'*W', C

        vision_feats = vision_feats.view(
            B // T if self.keep_time_dim else B, Npatch, *vision_feats.shape[1:]
        )
        vision_patch_labels = (
            vision_patch_labels.view(By, Npatch, *vision_patch_labels.shape[1:])
            if prepped_y is not None
            else None
        )
        vision_feats = vision_feats + self.camera_embeddings(
            torch.tensor([0, 1], device=vision_feats.device)
        ).unsqueeze(0).unsqueeze(-2)

        # Merge Npatch with vectors per image
        vision_feats = vision_feats.flatten(1, 2)
        if prepped_y is not None:
            vision_patch_labels = vision_patch_labels.flatten(1, 2)

        if self.masking_rate > 0.0 and self.training:
            mask = (
                torch.rand(B, vision_feats.shape[1], 1, device=vision_feats.device)
                > self.masking_rate
            )
            vision_feats = vision_feats * mask

        # Project to required dim (2048)
        if self.frozen_dict.get("projector", False):
            self.projector.eval()

        vision_feats = self.projector(vision_feats)

        if len(original_shape) == 5:
            if self.keep_time_dim:
                vision_feats = vision_feats.reshape(
                    original_shape[0], -1, *vision_feats.shape[1:]
                )
            else:
                vision_feats = vision_feats.reshape(
                    original_shape[0], original_shape[1], *vision_feats.shape[1:]
                )
            vision_patch_labels = (
                vision_patch_labels.reshape(
                    original_shape[0], 1, *vision_patch_labels.shape[1:]
                )
                if prepped_y is not None
                else None
            )
        if prepped_y is not None:
            return vision_feats, vision_patch_labels

        return vision_feats

    def decode(self, z):
        raise NotImplementedError

    def interpret(self, z):
        raise NotImplementedError
