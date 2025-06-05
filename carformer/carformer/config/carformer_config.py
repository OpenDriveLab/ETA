# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" CarFormer configuration, adapted from GPT-2 configuration """
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING

import os
import yaml
from typing import Union
from transformers import AutoConfig

logger = logging.get_logger(__name__)


class CarformerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Wanderer`] model or [`Ponderer`] model. It is used to
    instantiate the model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optionabackbone_configl*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "max_position_embeddings": "max_position_embeddings",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }

    def __init__(
        self,
        **kwargs,
    ):
        # import ipdb; ipdb.set_trace()
        # Load config from dict
        # import ipdb; ipdb.set_trace()
        # self.backbone_config = PretrainedConfig.from_dict(backbone)
        if "backbone" in kwargs:
            backbone = kwargs.pop("backbone")
            if "model_type" in backbone:
                backbone_cfg_cls = CONFIG_MAPPING[backbone["model_type"]]
                self.backbone = backbone_cfg_cls.from_dict(backbone)
            else:
                # import ipdb; ipdb.set_trace()
                self.backbone = AutoConfig.from_pretrained(
                    backbone["init_name_or_path"]
                )
        elif "init_name_or_path" in kwargs:
            self.backbone = AutoConfig.from_pretrained(kwargs["init_name_or_path"])
        # import ipdb; ipdb.set_trace()
        super().__init__(**kwargs)
        if hasattr(self, "n_embd"):
            self.hidden_size = self.n_embd

        if hasattr(self, "training"):
            if "use_future_vehicle_forcast" in self.training:
                print("Fixing typo in config")
                self.training["use_future_vehicle_forecast"] = self.training[
                    "use_future_vehicle_forcast"
                ]

        self.dotify()

    def dotify(self):
        # Convert all dict attributes to dotdict
        # from omegaconf.dictconfig import DictConfig
        from dotmap import DotMap as ddict

        # Iterate over all attributes
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = ddict(value, _dynamic=False)

    def dedotify(self):
        # Convert all dotdict attributes to dict
        # Iterate over all attributes
        from dotmap import DotMap as ddict

        for key, value in self.__dict__.items():
            if isinstance(value, ddict):
                self.__dict__[key] = dict(value)

    def __getattr__(self, attr: str, default=None):
        # Backward compatibility with old attribute names
        # Raise warning on each access to deprecated attributes
        # print(attr)
        if attr == "backbone":
            raise AttributeError(f"Attribute {attr} not found")
        # print(attr)

        if getattr(self, "backbone", None) is not None:
            # import warnings
            # print(f"attr {attr} not found, activating deprecated failsafe")
            # warnings.warn(
            #     f"Attribute {attr} will be deprecated in favor of backbone.{attr}. Please update your code.",
            #     DeprecationWarning,
            # )
            return getattr(self.backbone, attr)
        else:
            raise AttributeError(f"Attribute {attr} not found")

    @staticmethod
    def from_hydra(hydra_cfg: Any) -> "CarformerConfig":
        from omegaconf import OmegaConf

        return CarformerConfig.from_dict(
            OmegaConf.to_container(hydra_cfg, resolve=True)
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        # Call super to save config.json
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
