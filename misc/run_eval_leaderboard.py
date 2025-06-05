import sys
from leaderboard.leaderboard_evaluator_local import main
import argparse
import os
import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../Bench2DriveZoo/team_code/config", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg_org = cfg.copy()
    cfg = cfg.experiments

    print(cfg_org.eval.routes)
    print(cfg_org.checkpoint)
    print("Working directory : {}".format(os.getcwd()))
    print(f"Save gifs: {cfg_org.save_explainability_viz}")

    # create result folder
    Path(cfg_org.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg_org.CUDA_VISIBLE_DEVICES}"

    arg_dict0 = OmegaConf.to_container(cfg_org.eval, resolve=True)
    arg_dict1 = OmegaConf.to_container(cfg, resolve=True)
    arg_dict2 = OmegaConf.to_container(cfg_org, resolve=True)
    arg_dict1.update(arg_dict2)
    arg_dict1.update(arg_dict0)
    args = argparse.Namespace(**arg_dict1)

    from leaderboard import leaderboard_evaluator_local
    import numpy as np

    # np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)

    leaderboard_evaluator_local.main_eval(args)


if __name__ == "__main__":
    main()
