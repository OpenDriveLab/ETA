import os
import re
from .carformer_config import CarformerConfig

_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.,=-]")


# Adapted from https://tedboy.github.io/flask/_modules/werkzeug/utils.html#secure_filename
# Changed in order to work without importing flask
def sanitize_shorten_ckppath(path):
    path = path.replace("use_light_encoder", "ltenc")
    # ema_every_steps = getattr(self.ponderer.config.training, "ema_every_steps", 1)
    # ema_start = getattr(self.ponderer.config.training, "ema_start", 0)
    # ema_end_epoch = getattr(self.ponderer.config.training, "ema_end_epoch", -1)
    path = path.replace("gen_mask_for_action", "amsk")
    path = path.replace("turns", "trn")
    path = path.replace("optimizer", "opt")
    path = path.replace("ema_every_steps", "emaint")
    path = path.replace("ema_start", "emast")
    path = path.replace("ema_end_epoch", "emagnd")
    path = path.replace("backbone", "bb")
    path = path.replace("training.rgb_crop.crop_size", "trncrp")
    path = path.replace("training.ema_decay", "emadc")
    path = path.replace("training/", "")
    path = path.replace("bev/", "")
    path = path.replace("llava1pt6", "lv16")
    path = path.replace("rgb_front", "rgb")
    path = path.replace("enabled", "T")
    path = path.replace("deepspeed", "ds")
    path = path.replace("use_gt_frc_only", "gtfrconly")
    path = path.replace("combofrc", "cfrc")
    path = path.replace("ignore_past_for_length", "igpastln")
    path = path.replace("zero_out_frc_branch", "zfbrc")
    path = path.replace("use_gt_frc", "gtfrc")
    path = path.replace("use_past_horizon", "up")
    path = path.replace("backbone.", "")
    path = path.replace("optimizer.", "")
    path = path.replace("kwargs.", "")
    path = path.replace("weight_decay", "")
    path = path.replace("hyperparams.", "")
    path = path.replace("hyperparams/", "")
    path = path.replace("training.", "")
    path = path.replace("use_real_latent_ratio", "rltrtio")
    path = path.replace("past_horizon", "phrz")
    path = path.replace("llama-", "lma-")
    path = path.replace("micro", "mc")
    path = path.replace("mini", "mn")
    path = path.replace("future_horizon", "hrz")
    path = path.replace("future_horizon", "ftr")
    path = path.replace("ema_enabled", "ema")
    path = path.replace("use_predicted_latent_with_gap", "prdltnt")
    path = path.replace("bucket_weights.type", "bktype")
    path = path.replace("preferturns", "pturns")
    path = path.replace("prefer", "p")
    path = path.replace("+experiments", "exps")
    path = path.replace("num_epochs", "eps")
    path = path.replace("batch_size", "bs")
    path = path.replace("False", "F")
    path = path.replace("True", "T")
    path = path.replace("forecast_steps", "frc_steps")
    path = path.replace("loss_params.", "")
    path = path.replace("action", "actn")
    path = path.replace("forecast", "frc")
    path = path.replace("classification", "cls")
    path = path.replace("state", "stt")
    path = path.replace("dataset", "dts")
    path = path.replace("subsample_ratio", "smplrtio")
    path = path.replace("reconstruction", "rcns")
    path = path.replace("wandb_tag", "wnb")
    path = path.replace("gradient_accumulation_steps", "gacc")
    path = path.replace("dropout", "dp")
    path = path.replace("use_future_vehicle_forcast", "dofrc")
    path = path.replace("normalize_goal", "nrmgoal")
    path = path.replace("weighted_sampling", "wtsmpl")
    path = path.replace("use_future_vehicle_frc", "dofrc")
    path = path.replace("light_select_layer", "slclr")
    path = path.replace("use_light_encoder_backbone", "ltenc")
    path = path.replace("actn_gap", "agap")

    for sep in os.path.sep, os.path.altsep:
        if sep:
            path = path.replace(sep, " ")
    path = str(_filename_ascii_strip_re.sub("", "_".join(path.split()))).strip("._")
    path = path.replace("training_bev_rgb_backbonergb_backbone", "rgbbb")
    path = path.replace("training_goal", "gl")
    path = path.replace("dual_target_point", "2tp")
    path = path.replace("rgb_backbone", "rgbb")
    return path


def config_init():
    from dataclasses import dataclass

    from omegaconf import MISSING, OmegaConf

    import hydra
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    def merge_keys(*cfg):
        all_keys = set()
        for c in cfg:
            all_keys.update([str(x) for x in c.keys()])
        return "-".join(sorted(all_keys))

    # If has key, return True, else return False
    def has_key(cfg, key):
        return key in cfg

    def get_key(cfg, key, *args):
        # If args is not empty, recurse after getting the key
        if len(args) > 0:
            if key in cfg:
                try:
                    return get_key(cfg[key], *args)
                except KeyError:
                    return get_key(cfg[key], "reward")
                    raise KeyError(f"Key {args} not found in {cfg[key]}")
            else:
                raise KeyError(f"Key {key} not found in {cfg}")

        if key in cfg:
            return cfg[key]
        else:
            raise KeyError(f"Key {key} not found in {cfg}")

    def resolve_quantizer_path(keys, data_format, quantizer_dict):
        # quantizer_key = "plant" if plant_data else "legacy"
        if data_format == "plant":
            quantizer_key = "plant"
        else:
            quantizer_key = "legacy"
        return get_key(quantizer_dict, quantizer_key, keys)

    OmegaConf.register_new_resolver("merge_keys", merge_keys)

    OmegaConf.register_new_resolver("has_key", has_key)

    OmegaConf.register_new_resolver("get_key", get_key)
    OmegaConf.register_new_resolver("eval", eval)

    bool_to_str = lambda x: "true" if x else "false"
    OmegaConf.register_new_resolver("bool_to_str", bool_to_str)
    OmegaConf.register_new_resolver("resolve_quantizer_path", resolve_quantizer_path)
    OmegaConf.register_new_resolver("sanitize", sanitize_shorten_ckppath)
