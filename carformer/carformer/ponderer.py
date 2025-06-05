import torch
import torch.nn.functional as F
from carformer.utils import (
    TokenTypeIDs,
    deinterleave,
    fuzzy_extract_state_dict_from_checkpoint,
)
from carformer.config import CarformerConfig
from carformer.perception.rgb import RGBEncoder
import os
from transformers import AutoModel, PreTrainedModel


# Ponderer model
class Ponderer(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Ponderer, self).__init__(config)
        self.cfg = config
        # TODO: Possibly decouple the backbone from the Ponderer model config better (e.g. config.backbone_config)
        if hasattr(config, "backbone"):
            self.backbone_config = config.backbone
        else:
            self.backbone_config = config
        # Must be before GPT2 init because vocab size is set there
        self.init_model()

        if hasattr(self.cfg, "init_from_lm_ckp") and self.cfg.init_from_lm_ckp:
            #     from transformers import AutoModel
            # Ensure that the initialization config is the same as the current model
            from transformers import AutoConfig

            weight_model_config = AutoConfig.from_pretrained(self.cfg.init_name_or_path)

            for key in self.backbone_config.keys():
                if key in weight_model_config:
                    assert (
                        weight_model_config[key] == self.backbone_config[key]
                    ), f"Key {key} does not match. {weight_model_config[key]} != {self.backbone_config[key]}"

            # model_weights = AutoModel.from_pretrained(self.cfg.init_name_or_path).state_dict()
            self.backbone = AutoModel.from_pretrained(self.cfg.init_name_or_path)
        else:
            self.backbone = AutoModel.from_config(self.backbone_config)

        if (
            hasattr(self.cfg, "gradient_checkpointing")
            and self.cfg.gradient_checkpointing
        ):
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

        if self.cfg.training.get("use_gt_frc_only", False):
            assert self.cfg.training.get("use_gt_frc", False)

    def forward(
        self, input_dict, calculate_loss=True, return_labels=False, return_inputs=False
    ):
        backbone_inputs = self.prepare_inputs(
            input_dict,
            create_autoregressive_labels=calculate_loss,
            calculate_loss=calculate_loss,
        )

        # if calculate_loss:
        backbone_inputs, labels = backbone_inputs
        backbone_outputs = self.backbone(**backbone_inputs)

        label_type_ids = labels["input_ids"]

        deinterleaved_outputs = deinterleave(
            backbone_outputs["last_hidden_state"], label_type_ids
        )

        actions_pred = None
        loss = None

        # Inputs to the decoding function
        decoder_inputs = {
            TokenTypeIDs.ACTION: deinterleaved_outputs[TokenTypeIDs.ACTION],
        }

        if self.cfg.training["gen_masks_for_action"] and calculate_loss:
            if self.cfg.training.get("utilize_fast_current_latent", False):
                bev_token_count = labels["original_rgb_latent_length"]
            else:
                bev_token_count = None
            decoder_inputs[TokenTypeIDs.BEV] = {
                "bev": deinterleaved_outputs[TokenTypeIDs.BEV][:, :bev_token_count],
                "action": deinterleaved_outputs[TokenTypeIDs.ACTION],
            }

            if "bev_mask" in labels:
                bev_mask_size = labels["bev_mask"].shape[-1]
                decoder_inputs[TokenTypeIDs.BEV]["bev"] = decoder_inputs[
                    TokenTypeIDs.BEV
                ]["bev"][:, -bev_mask_size:]

        decoded_outputs = self.decode(decoder_inputs)

        if TokenTypeIDs.ACTION in decoded_outputs:
            actions_pred = decoded_outputs[TokenTypeIDs.ACTION]

        if TokenTypeIDs.BEV in decoded_outputs:
            bev_pred_mask = decoded_outputs[TokenTypeIDs.BEV]
        else:
            bev_pred_mask = None

        predictions = {
            "action": actions_pred,
            "bev": bev_pred_mask,
            "deinterleaved_outputs": deinterleaved_outputs,
        }

        if (
            self.cfg.training["use_future_vehicle_forecast"]
            and not self.cfg.training.get("use_predicted_latent_with_gap", False)
            and calculate_loss
        ):

            if "original_rgb_latent_length" in labels:
                bev_token_count = labels["original_rgb_latent_length"]
            else:
                bev_token_count = None

            forecasting_inputs = {
                "bev": deinterleaved_outputs[TokenTypeIDs.BEV][:, :bev_token_count],
                "action": decoded_outputs[TokenTypeIDs.ACTION],
            }

            if "bev_mask" in labels:
                bev_mask_size = labels["bev_mask"].shape[-1]
                forecasting_inputs["bev"] = forecasting_inputs["bev"][
                    :, -bev_mask_size:
                ]

            forecast_rgb = self.forecast_bev(forecasting_inputs)

            predictions["forecast_rgb"] = forecast_rgb
        else:
            if "original_rgb_latent_length" in labels:
                bev_token_count = labels["original_rgb_latent_length"]
            else:
                bev_token_count = None
            if calculate_loss:
                forecast_rgb = labels["input_bev_latent"][:, :bev_token_count]
                predictions["forecast_rgb"] = forecast_rgb

                if "bev_mask" in labels:
                    bev_mask_size = labels["bev_mask"].shape[-1]
                    predictions["forecast_rgb"] = predictions["forecast_rgb"][
                        :, -bev_mask_size:
                    ]

                if self.cfg.training.get("vae_target_supervision", False):
                    predictions["forecast_rgb"] = self.bev_vae_proj(
                        predictions["forecast_rgb"]
                    )

        return_values = [predictions]

        if calculate_loss:
            loss = self.calculate_loss(
                input_dict,
                backbone_inputs,
                labels,
                deinterleaved_outputs,
                predictions,
            )
            return_values.append(loss)

        if return_labels:
            return_values.append(labels)

        if return_inputs:
            return_values.append(backbone_inputs)

        return tuple(return_values) if len(return_values) > 1 else return_values[0]

    def decode(self, output_dict):
        """
        Decode a sequence of tokens given an output dictionary.
        Args:
            output_dict (dict): Dictionary of output token ids, indexed by token type
        Returns:
            output_dict (dict): Dictionary of output values, indexed by token type
        """

        for token_type_id, decoder in [
            (TokenTypeIDs.ACTION, self.action_decoder),
            (TokenTypeIDs.BEV, self.bev_decoder),
        ]:
            if token_type_id in output_dict:
                output_dict[token_type_id] = decoder(output_dict[token_type_id])

        return output_dict

    def prepare_inputs(
        self, input_dict, create_autoregressive_labels=False, calculate_loss=True
    ):
        """
        Prepare inputs for the backbone model.
        Args:
            input_dict (dict): Dictionary of inputs
            create_autoregressive_labels (bool): Whether to prepare labels for self-supervised autoregressive decoding or not
            calculate_loss (bool): Whether to calculate loss or not. Essentially whether or not we are at test time inference.
        Returns:
            backbone_inputs (dict): Dictionary of inputs for the backbone model
            labels (dict): Dictionary of labels for the backbone model (if create_autoregressive_labels is True)
        """
        input_dict = {k: v for k, v in input_dict.items()}
        # Autocast inputs if not casted,
        dtype_to_cast = next(self.state_encoder.parameters()).dtype

        # Cast any fp inputs to dtype_to_cast
        for key, value in list(input_dict.items()):
            if torch.is_floating_point(value) and not dtype_to_cast == value.dtype:
                input_dict[key] = value.to(dtype_to_cast)

        if self.cfg.training.get("augment_data", False) and self.training:
            raise NotImplementedError("Dynamic data augmentation not yet supported")

        input_dict = self.preprocess_input_dict(input_dict)

        goals = None
        if self.cfg.training["condition_on_goal"]:
            goals = input_dict["goal"]  # (batch_size, num_steps, goal_size)

        states = None
        if "state" in input_dict:
            states = input_dict["state"]  # (batch_size, num_steps, state_size)
        bev = input_dict[
            "bev_latent"
        ]  # (batch_size, num_steps, npatches, bev_latent_dim)
        bev = bev.reshape(bev.shape[0], -1, bev.shape[-1])

        if self.cfg.training.get("use_real_latent_ratio", 0.0) > 0.0 and self.training:
            real_latent = input_dict["target_rgb_latent"]
            real_latent = real_latent.reshape(
                real_latent.shape[0], -1, real_latent.shape[-1]
            )  # Batch_size, num_steps*npatches, latent_dim

            # Mask of size B, whether to use real latent or not
            use_real_latent = (
                torch.rand(bev.shape[0], device=bev.device)
                < self.cfg.training["use_real_latent_ratio"]
            )

            bev = torch.where(use_real_latent.reshape(-1, 1, 1), real_latent, bev)
        if self.cfg.training.get("use_predicted_latent_with_gap", False):
            if self.cfg.training.get("zero_out_frc_branch", False):
                # print("REPLACING PREDICTED LATENT WITH ZEROES")
                bev = torch.zeros_like(bev)

            if self.cfg.training.get("use_gt_frc", False):
                # print("REPLACING PREDICTED LATENT WITH GT")
                bev = input_dict["target_rgb_latent"]

            if self.cfg.training.get("utilize_fast_current_latent", False):
                current_fast_latent = input_dict["current_rgb_latent"]
                current_fast_latent = current_fast_latent.reshape(
                    current_fast_latent.shape[0], -1, current_fast_latent.shape[-1]
                )  # Batch_size, num_steps*npatches, latent_dim

                if self.cfg.training.get(
                    "use_gt_frc_only", False
                ) and self.cfg.training.get("use_gt_frc", False):
                    # fmt:skip
                    bev = bev
                elif self.cfg.training.get("use_light_as_query", False):
                    bev = bev  # Use light only as query
                else:
                    bev = torch.cat([bev, current_fast_latent], dim=1)

        widths = {}
        widths["bev"] = bev.shape[1]

        # Encode
        # Note: This does not seem incremental decoding friendly
        #       Might need reworking or modification
        if self.cfg.training["condition_on_goal"]:
            goals = self.goal_encoder(goals).to(bev.device)
            widths["goal"] = goals.shape[-1]


        states = self.state_encoder(states)  # (batch_size, num_steps, n_embd)
        action_queries = torch.arange(30, dtype=torch.long, device=states.device)
        action_queries = self.action_queries(action_queries)
        action_queries = action_queries.unsqueeze(0).expand(
            states.shape[0], *action_queries.shape
        )  # Expand to size of states
        widths["state"] = 1
        widths["action"] = 30

        # Interleave the inputs to get the sequence as:
        # [goal0, state0, action0, reward0, goal1, state1, action1, reward1, ...]
        # This is the format that the backbone expects
        # If config.goal_conditioning is "global", then the goal is not interleaved and is appended to the beginning of the sequence as a prefix
        if goals is None or self.cfg.training["goal_conditioning_type"] == "global":
            raise NotImplementedError
        elif self.cfg.training["goal_conditioning_type"] == "local":
            # Concatenate the goals to the beginning of every timestep
            inputs = torch.cat([bev, goals, states, action_queries], dim=1)
        else:
            raise ValueError(
                "Invalid goal_conditioning value: {}. Either disable goal conditioning or use a valid goal conditioning type".format(
                    self.cfg.training["goal_conditioning_type"]
                )
            )

        input_ids = torch.cat(
            [
                torch.ones((1, bev.shape[1]), dtype=torch.long) * TokenTypeIDs.BEV,
                torch.ones((1, goals.shape[1]), dtype=torch.long) * TokenTypeIDs.GOAL,
                torch.ones((1, states.shape[1]), dtype=torch.long) * TokenTypeIDs.STATE,
                torch.ones((1, action_queries.shape[1]), dtype=torch.long)
                * TokenTypeIDs.ACTION,
            ],
            dim=1,
        ).to(states.device)

        # Expand dim 0 to batch size
        input_ids = input_ids.expand(states.shape[0], *input_ids.shape[1:])

        # output_labels = actions

        result_dict = {}

        result_dict["inputs_embeds"] = inputs

        labels = {}
        labels["input_ids"] = input_ids
        labels["input_bev_latent"] = bev

        if "original_rgb_latent_length" in input_dict:
            labels["original_rgb_latent_length"] = input_dict[
                "original_rgb_latent_length"
            ]

        if create_autoregressive_labels:
            actions = input_dict["action"]  # (batch_size, num_steps, action_size)
            labels["action"] = actions
            # rewards = input_dict["reward"]  # (batch_size, num_steps, reward_size)
            if "mask_labels" in input_dict:
                mask_labels = input_dict["mask_labels"]
                if mask_labels is not None:
                    labels["bev_mask"] = mask_labels.flatten(1, 2)

            if "target_rgb_latent" in input_dict:
                labels["target_rgb_latent"] = input_dict["target_rgb_latent"].detach()
                labels["target_bev"] = input_dict["target_rgb_front"]

        return result_dict, labels

    def preprocess_input_dict(self, input_dict, calculate_loss=True):
        """
        Preprocess the input dictionary by performing deterministic operations that are the same everytime.
        Every run of this function with the same input_dict should return the same output_dict.
        This way, we can cache the output_dict and reuse it for multiple runs, or even delegate it to the dataloader.
        Args:
            input_dict (dict): Dictionary of inputs
            calculate_loss (bool): Whether to calculate loss or not. Essentially whether or not we are at test time inference.
        Returns:
            input_dict (dict): Dictionary of inputs
        """
        if self.cfg.training["object_level"]:
            if self.cfg.training.get("use_slots", False):
                raise NotImplementedError
        else:
            if "rgb_latent" not in input_dict:
                rgb = input_dict["rgb_front"]

                if self.cfg.training.get("utilize_fast_current_latent", False):
                    assert "target_rgb_front" in input_dict
                    input_dict["current_rgb_latent"] = self.light_bev_encoder(
                        input_dict["target_rgb_front"]
                    )

                if "mask" in input_dict:
                    mask = input_dict["mask"]
                    bev_latent, mask_labels = self.bev_encoder(rgb, mask)
                else:
                    bev_latent = self.bev_encoder(rgb)
                    mask_labels = None

                if self.cfg.training.get("use_predicted_latent_with_gap", False):
                    forecast_input_dict = {
                        "bev": bev_latent,
                        "bev_light": (
                            input_dict["current_rgb_latent"]
                            if "current_rgb_latent" in input_dict
                            else None
                        ),
                        "frc_wps": input_dict["frc_wps"],
                        "frc_speed": input_dict["frc_speed"],
                        "frc_goal": input_dict["frc_goal"],
                    }
                    # import ipdb

                    # ipdb.set_trace()
                    orig_bev_latent = bev_latent
                    bev_latent = self.forecast_bev(forecast_input_dict)
                    input_dict["orig_bev_latent"] = orig_bev_latent

                input_dict["bev_latent"] = bev_latent
                input_dict["mask_labels"] = mask_labels

                if self.cfg.training.get("utilize_fast_current_latent", False):
                    input_dict["original_rgb_latent_length"] = (
                        input_dict["bev_latent"].shape[1]
                        * input_dict["bev_latent"].shape[2]
                    )

            if "target_rgb_front" in input_dict:
                if self.cfg.training.get("use_gt_frc", False):
                    assert (
                        self.cfg.training.get("vae_target_supervision", None) is None
                    ), "Cannot use GT forecast with VAE supervision"
                    input_dict["target_rgb_latent"] = self.bev_encoder(
                        input_dict["target_rgb_front"], ema=True
                    ).flatten(1, 2)
                else:
                    if calculate_loss:
                        with torch.no_grad():
                            if self.cfg.training.get("vae_target_supervision", False):
                                input_dict["target_rgb_latent"] = self.bev_encoder(
                                    input_dict["target_rgb_front"], vqvae=True
                                )  # TODO: Make it so it requires flattening again
                            else:
                                input_dict["target_rgb_latent"] = self.bev_encoder(
                                    input_dict["target_rgb_front"], ema=True
                                ).flatten(1, 2)

        return input_dict

    def get_preprocessed_cache_parametrized_dirname(self):
        """
        Get a unique preprocessing directory name that is unique to the parameters used in preprocessing, such
        as the bev encoder and checkpoint path.
        """
        return "dummy"

    def calculate_loss(
        self, input_dict, inputs, labels, deinterleaved_outputs, predictions
    ):
        # Calculate loss
        loss_dict = {}
        loss = 0

        if (
            TokenTypeIDs.ACTION in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.ACTION].numel() > 0
        ):
            wps = predictions["action"]["wps"]
            path = predictions["action"]["path"]

            wp_label = labels["action"][:, :, 20:].flatten(1, 2)
            path_label = labels["action"][:, :, :20].flatten(1, 2)

            B = wps.shape[0]
            wp_label = torch.diff(
                wp_label,
                prepend=torch.zeros(1).to(wp_label).reshape(1, 1, 1).expand(B, 1, 2),
                dim=-2,
            )
            path_label = torch.diff(
                path_label,
                prepend=torch.zeros(1).to(path_label).reshape(1, 1, 1).expand(B, 1, 2),
                dim=-2,
            )

            loss_dict["wp_loss"] = F.mse_loss(wps, wp_label)

            loss += (loss_dict["wp_loss"]) * self.cfg.training["loss_params"]["wp_loss"]

            loss_dict["path_loss"] = F.mse_loss(path, path_label)

            loss += (
                loss_dict["path_loss"] * self.cfg.training["loss_params"]["path_loss"]
            )

        if "bev" in predictions and "bev_mask" in labels:
            bev_pred = predictions["bev"]
            bev_labels = (labels["bev_mask"] > 0.5).to(bev_pred)

            loss_dict["image_loss"] = F.binary_cross_entropy_with_logits(
                bev_pred, bev_labels, reduction="mean", reduce=True
            )

            loss += (
                loss_dict["image_loss"] * self.cfg.training["loss_params"]["mask_loss"]
            )

        if "forecast_rgb" in predictions and "target_rgb_latent" in labels:
            forecast_rgb = predictions["forecast_rgb"]
            target_rgb_latent = labels["target_rgb_latent"]

            if (
                self.cfg.training.get("vae_target_supervision", False)
                and not self.bev_encoder.autoencode_continuous
            ):
                loss_dict["forecast_loss"] = F.cross_entropy(
                    forecast_rgb.flatten(0, -2), target_rgb_latent.flatten()
                )
            else:
                loss_dict["forecast_loss"] = F.l1_loss(forecast_rgb, target_rgb_latent)

            loss += (
                loss_dict["forecast_loss"]
                * self.cfg.training["loss_params"]["state_forecast"]
            )

        loss_dict["loss"] = loss
        return loss_dict

    def init_model(self):
        self.quantization_offset_map = {}
        self.quantization_vocab_size_map = {}
        self.cfg.quantization_vocab_size_map = self.quantization_vocab_size_map
        self.cfg.quantization_offset_map = self.quantization_offset_map
        self.use_slots = False
        self.normalize_goal = self.cfg.training.get("normalize_goal", False)

        self.bev_encoder = RGBEncoder(self.cfg.training["rgb_backbone"])

        if self.cfg.training.get("use_predicted_latent_with_gap", False):
            if self.cfg.training.get("utilize_fast_current_latent", False):
                self.light_bev_encoder = RGBEncoder(
                    self.cfg.training["light_rgb_backbone"]
                )

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.backbone_config.hidden_size),
        )

        if self.cfg.training["goal_type"] == "target_point":
            if "mean" in self.cfg.training["goal"]["target_point"]:
                goal_mean = self.cfg.training["goal"]["target_point"]["mean"]
                goal_std = self.cfg.training["goal"]["target_point"]["std"]
                # Register as buffer
                self.register_buffer("goal_mean", torch.tensor(goal_mean))
                self.register_buffer("goal_std", torch.tensor(goal_std))

                self.goal_projector = torch.nn.Sequential(
                    torch.nn.Linear(2, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, self.backbone_config.hidden_size),
                )

                def goal_encoder(goals):
                    if self.normalize_goal:
                        goals = (goals - self.goal_mean) / self.goal_std
                    return self.goal_projector(goals)

                self.goal_encoder = goal_encoder
            else:
                self.goal_encoder = torch.nn.Sequential(
                    torch.nn.Linear(2, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, self.backbone_config.hidden_size),
                )
        else:
            if "std" in self.cfg.training["goal"]["dual_target_point"]:
                goal_mean = self.cfg.training["goal"]["dual_target_point"]["mean"]
                goal_std = self.cfg.training["goal"]["dual_target_point"]["std"]
            else:
                goal_mean = [[0.0, 0.0], [0.0, 0.0]]
                goal_std = [[1.0, 1.0], [1.0, 1.0]]

            self.register_buffer("goal_mean", torch.tensor(goal_mean))
            self.register_buffer("goal_std", torch.tensor(goal_std))

            self.goal_projector = torch.nn.Sequential(
                torch.nn.Linear(2, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, self.backbone_config.hidden_size),
            )

            def goal_encoder(goals):
                # print(goals)
                if self.normalize_goal:
                    goals = (goals - self.goal_mean) / self.goal_std
                # print(goals)
                return self.goal_projector(goals).flatten(1, 2)

            self.goal_encoder = goal_encoder

        self.action_queries = torch.nn.Embedding(
            30, self.backbone_config.hidden_size
        )  # 10 for WP, 20 for path

        self.wp_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_config.hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        self.path_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_config.hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

        def decode_actions(latents):
            wp_latents = latents[:, :10]
            path_latents = latents[:, 10:]

            wps = self.wp_mlp(wp_latents)
            path = self.path_mlp(path_latents)

            return {"wps": wps, "path": path}

        def decode_bev(input_dict):
            bev_latents = input_dict["bev"]
            action_latents = input_dict["action"]

            bev_pred = torch.einsum("BNH,BDH->BND", bev_latents, action_latents).mean(
                -1
            )

            return bev_pred

        if self.cfg.training["use_future_vehicle_forecast"]:
            if not self.cfg.training.get("use_predicted_latent_with_gap", False):
                layers = []

                use_post_mlp = self.cfg.training.get("pred_latent_post_mlp", False)
                pred_latent_ffn_dropout = self.cfg.training.get(
                    "pred_latent_ffn_dropout", 0.1
                )
                pred_latent_layers = self.cfg.training.get("pred_latent_layers", 2)
                pred_latent_ffn_hidden = self.cfg.training.get(
                    "pred_latent_ffn_hidden", 2048
                )

                layers.append(
                    torch.nn.Linear(
                        (
                            self.backbone_config.hidden_size
                            * (self.cfg.training["past_horizon"] + 1)
                        )
                        + 20,
                        self.backbone_config.hidden_size,
                    )
                )

                layers.append(
                    torch.nn.TransformerEncoder(
                        torch.nn.TransformerEncoderLayer(
                            d_model=self.backbone_config.hidden_size,
                            nhead=8,
                            dim_feedforward=pred_latent_ffn_hidden,
                            dropout=pred_latent_ffn_dropout,
                            activation="gelu",
                        ),
                        num_layers=pred_latent_layers,
                    )
                )

                if use_post_mlp:
                    layers.append(
                        torch.nn.Linear(
                            self.backbone_config.hidden_size,
                            self.backbone_config.hidden_size,
                            bias=False,
                        )
                    )

                self.bev_forecast_module = torch.nn.Sequential(*layers)

                def forecast_bev(input_dict):
                    bev_latents = input_dict["bev"]
                    action_latents = input_dict["action"]["wps"].flatten(1)

                    frc_inputs = torch.cat(
                        [
                            bev_latents,
                            action_latents.unsqueeze(1).expand(
                                -1, bev_latents.shape[1], -1
                            ),
                        ],
                        dim=-1,
                    )

                    return self.bev_forecast_module(frc_inputs)

            else:
                if self.cfg.training.get("use_light_as_query", False):
                    p_horizon = self.cfg.training["past_horizon"]
                    print(f"Past horizon: {p_horizon}\n" * 20)
                    layers = []

                    use_post_mlp = self.cfg.training.get("pred_latent_post_mlp", False)
                    pred_latent_ffn_dropout = self.cfg.training.get(
                        "pred_latent_ffn_dropout", 0.1
                    )
                    pred_latent_layers = self.cfg.training.get("pred_latent_layers", 2)
                    pred_latent_ffn_hidden = self.cfg.training.get(
                        "pred_latent_ffn_hidden", 2048
                    )
                    pred_latent_use_metadata = self.cfg.training.get(
                        "pred_latent_use_metadata", True
                    )

                    layers.extend(
                        [
                            torch.nn.Linear(
                                (
                                    self.backbone_config.hidden_size
                                    * (
                                        (self.cfg.training["past_horizon"] + 1)
                                        if not (
                                            "hiera"
                                            in self.cfg.training["rgb_backbone"][
                                                "model_path_rel"
                                            ]
                                        )
                                        else 1
                                    )
                                )
                                + (9 if pred_latent_use_metadata else 0),
                                self.backbone_config.hidden_size,
                            ),  # 2 WPS: 4, speed: 1, 2 TP: 4.
                            torch.nn.GELU(),
                            torch.nn.Linear(
                                self.backbone_config.hidden_size,
                                self.backbone_config.hidden_size,
                            ),
                        ]
                    )

                    self.bev_pre_projector = torch.nn.Sequential(*layers)

                    self.bev_pred_module = torch.nn.TransformerDecoder(
                        torch.nn.TransformerDecoderLayer(
                            d_model=self.backbone_config.hidden_size,
                            nhead=8,
                            dim_feedforward=pred_latent_ffn_hidden,
                            dropout=pred_latent_ffn_dropout,
                            activation="gelu",
                        ),
                        num_layers=pred_latent_layers,
                    )

                    layers = []
                    if use_post_mlp:
                        layers.append(
                            torch.nn.Linear(
                                self.backbone_config.hidden_size,
                                self.backbone_config.hidden_size,
                                bias=False,
                            )
                        )

                        self.bev_post_projector = torch.nn.Sequential(*layers)
                    else:
                        self.bev_post_projector = torch.nn.Identity()

                    def forecast_bev(input_dict):
                        bev_latents = input_dict["bev"]
                        bev_light_latents = input_dict["bev_light"]

                        orig_bev_shape = bev_latents.shape

                        orig_bev_shape = tuple(
                            [orig_bev_shape[0], 1, *orig_bev_shape[2:]]
                        )

                        action_latents = input_dict["frc_wps"][:, :, :2, :].flatten(1)
                        speed_latents = input_dict["frc_speed"].flatten(1)
                        goals = input_dict["frc_goal"]
                        if self.normalize_goal:
                            goals = (goals - self.goal_mean) / self.goal_std
                        tp_latents = goals.flatten(1)

                        bev_latents = bev_latents.permute(0, 2, 1, 3).flatten(2, 3)
                        bev_light_latents = bev_light_latents.permute(
                            0, 2, 1, 3
                        ).flatten(2, 3)

                        if pred_latent_use_metadata:
                            frc_inputs = torch.cat(
                                [
                                    bev_light_latents,
                                    action_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                    speed_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                    tp_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                ],
                                dim=-1,
                            )
                        else:
                            frc_inputs = bev_light_latents

                        # Queries: Bev_light_latents
                        # Keys, Values: Bev_latents
                        bev_pred = self.bev_pred_module(
                            self.bev_pre_projector(frc_inputs),
                            bev_latents,
                        )
                        return self.bev_post_projector(bev_pred).reshape(orig_bev_shape)

                else:
                    p_horizon = self.cfg.training["past_horizon"]
                    layers = []

                    use_post_mlp = self.cfg.training.get("pred_latent_post_mlp", False)
                    pred_latent_ffn_dropout = self.cfg.training.get(
                        "pred_latent_ffn_dropout", 0.1
                    )
                    pred_latent_layers = self.cfg.training.get("pred_latent_layers", 2)
                    pred_latent_ffn_hidden = self.cfg.training.get(
                        "pred_latent_ffn_hidden", 2048
                    )
                    pred_latent_use_metadata = self.cfg.training.get(
                        "pred_latent_use_metadata", True
                    )

                    layers.append(
                        torch.nn.Linear(
                            (
                                self.backbone_config.hidden_size
                                * (
                                    (self.cfg.training["past_horizon"] + 1)
                                    if not (
                                        "hiera"
                                        in self.cfg.training["rgb_backbone"][
                                            "model_path_rel"
                                        ]
                                    )
                                    else 1
                                )
                            )
                            + (9 if pred_latent_use_metadata else 0),
                            self.backbone_config.hidden_size,
                        )  # 2 WPS: 4, speed: 1, 2 TP: 4.
                    )

                    layers.append(
                        torch.nn.TransformerEncoder(
                            torch.nn.TransformerEncoderLayer(
                                d_model=self.backbone_config.hidden_size,
                                nhead=8,
                                dim_feedforward=pred_latent_ffn_hidden,
                                dropout=pred_latent_ffn_dropout,
                                activation="gelu",
                            ),
                            num_layers=pred_latent_layers,
                        )
                    )

                    if use_post_mlp:
                        layers.append(
                            torch.nn.Linear(
                                self.backbone_config.hidden_size,
                                self.backbone_config.hidden_size,
                                bias=False,
                            )
                        )
                    self.bev_forecast_module = torch.nn.Sequential(*layers)

                    def forecast_bev(input_dict):
                        # for k, v in input_dict.items():
                        #     print(k, v.shape)

                        bev_latents = input_dict["bev"]
                        orig_bev_shape = bev_latents.shape

                        orig_bev_shape = tuple(
                            [orig_bev_shape[0], 1, *orig_bev_shape[2:]]
                        )

                        action_latents = input_dict["frc_wps"][:, :, :2, :].flatten(1)
                        speed_latents = input_dict["frc_speed"].flatten(1)
                        goals = input_dict["frc_goal"]
                        if self.normalize_goal:
                            goals = (goals - self.goal_mean) / self.goal_std
                        tp_latents = goals.flatten(1)
                        # print(bev_latents.shape, action_latents.shape, speed_latents.shape, tp_latents.shape)
                        # bev_latents = bev_latents.flatten(1, 2)
                        bev_latents = bev_latents.permute(0, 2, 1, 3).flatten(2, 3)

                        if pred_latent_use_metadata:
                            frc_inputs = torch.cat(
                                [
                                    bev_latents,
                                    action_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                    speed_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                    tp_latents.unsqueeze(1).expand(
                                        -1, bev_latents.shape[-2], -1
                                    ),
                                ],
                                dim=-1,
                            )
                        else:
                            frc_inputs = bev_latents
                        # import ipdb

                        # ipdb.set_trace()
                        return self.bev_forecast_module(frc_inputs).reshape(
                            orig_bev_shape
                        )

            if self.cfg.training.get("vae_target_supervision", False):
                if self.bev_encoder.autoencode_continuous:
                    layers = [
                        torch.nn.Linear(
                            self.backbone_config.hidden_size,
                            self.backbone_config.hidden_size,
                        ),  # GELU
                        torch.nn.GELU(),
                        torch.nn.Linear(self.backbone_config.hidden_size, 512),  # GELU
                        torch.nn.GELU(),
                        torch.nn.Linear(512, 32),
                    ]
                    self.bev_vae_proj = torch.nn.Sequential(*layers)
                else:
                    layers = [
                        torch.nn.Linear(
                            self.backbone_config.hidden_size,
                            self.backbone_config.hidden_size,
                        ),  # GELU
                        torch.nn.GELU(),
                        torch.nn.Linear(self.backbone_config.hidden_size, 2048),  # GELU
                        torch.nn.GELU(),
                        torch.nn.Linear(2048, 2048),
                        torch.nn.GELU(),
                        torch.nn.Linear(
                            2048, self.bev_encoder.vqvae.config.codebook_size * 4
                        ),
                    ]

                    self.bev_vae_proj_layers = torch.nn.Sequential(*layers)

                    def bev_vae_proj_sample(bev_latents):
                        latents = self.bev_vae_proj_layers(bev_latents)
                        latents = latents.reshape(
                            *latents.shape[:-1],
                            4,
                            self.bev_encoder.vqvae.config.codebook_size,
                        )
                        # Pick the maximum scoring index from the codebook
                        return latents

                    self.bev_vae_proj = bev_vae_proj_sample
        else:
            forecast_bev = lambda x: None

        self.action_decoder = decode_actions
        self.bev_decoder = decode_bev
        self.forecast_bev = forecast_bev

        self.loss_params = self.cfg.training["loss_params"]

    def get_label_type_ids_from_tokens(self, tokens):
        raise NotImplementedError("This method is not implemented in Ponderer.")

    @staticmethod
    def from_pretrained(path, epoch=30, deepspeed="auto"):
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

        import inspect, carformer

        carformer_path = os.path.dirname(inspect.getfile(carformer))
        # Go one layer up
        carformer_path = os.path.dirname(carformer_path)

        # Replace encoder path
        if "rgb_backbone" in config.training:
            if config.training.rgb_backbone.frozen or not (
                "model_config" in config.training.rgb_backbone
            ):
                config.training.rgb_backbone.model_path = None
                # config.training.rgb_backbone.model_path = os.path.join(
                #     carformer_path, config.training.rgb_backbone.model_path_rel
                # )

                # if config.training.rgb_backbone.get("ema_enabled", False):
                #     config.training.rgb_backbone.ema_enabled = False
            else:
                # Do not load path, the weights and config should already exist.
                config.training.rgb_backbone.model_path = None

        if "light_rgb_backbone" in config.training:
            if config.training.light_rgb_backbone.frozen or not (
                "model_config" in config.training.light_rgb_backbone
            ):
                config.training.light_rgb_backbone.model_path = None
                # config.training.light_rgb_backbone.model_path = os.path.join(
                #     carformer_path, config.training.light_rgb_backbone.model_path_rel
                # )
            else:
                # Do not load path, the weights and config should already exist.
                config.training.light_rgb_backbone.model_path = None

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

            checkpoint = fuzzy_extract_state_dict_from_checkpoint(checkpoint)

            model.load_state_dict(checkpoint, strict=True)

        return model

    def ema_update(self, model, model_ema, device=None, beta=0.992):
        with torch.no_grad():
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                # TODO: use prefiltering for efficiency
                # params_to_fetch = _z3_params_to_fetch([param, param_ema
                #                                     ]) if zero_stage_3 else []
                # should_gather_param = len(params_to_fetch) > 0
                # with deepspeed.zero.GatheredParameters(
                #         params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
