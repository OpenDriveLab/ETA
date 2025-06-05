import numpy as np
import torch
import os
import json
from os.path import join
from .data_parser import Parser
import hashlib
from glob import glob
from .data_utils import (
    get_rgb_preprocessing_function_from_config,
    transform_waypoints,
    extract_forecast_targets,
)
from carformer.visualization.visutils import (
    point_to_canvas_coordinates_rel_to_center,
    draw_points_on_camera,
)
from collections import defaultdict


class DictCountWrapper(dict):
    def __init__(self, *args, **kwargs):
        super(DictCountWrapper, self).__init__(*args, **kwargs)
        self.access_counter = defaultdict(int)

    def __getitem__(self, key):
        self.access_counter[key] += 1
        return super(DictCountWrapper, self).__getitem__(key)


class Timestep:
    # @profile
    def __init__(
        self,
        parser,
        index,
        preprocessing_functions=None,
        filtering_functions=None,
        include_noise=True,
        wrap_dict=True,
        skip_keys_dict=None,
    ):
        self.index = index
        self.parser = parser

        self.goal = self.parser.get_goal(
            self.index, skip_keys=skip_keys_dict.get("goal", None)
        )
        self.state = self.parser.get_state(
            self.index,
            preprocessing_functions=preprocessing_functions,
            filtering_functions=filtering_functions,
            skip_keys=skip_keys_dict.get("state", None),
        )

        if include_noise:
            self.action, self.noisy = self.parser.get_action(
                self.index,
                include_noise=True,
                skip_keys=skip_keys_dict.get("action", None),
            )
        else:
            self.action = self.parser.get_action(
                self.index, skip_keys=skip_keys_dict.get("action", None)
            )

        self.reward = self.parser.get_reward(
            self.index, skip_keys=skip_keys_dict.get("reward", None)
        )

        if wrap_dict:
            self.goal = DictCountWrapper(self.goal)
            self.state = DictCountWrapper(self.state)
            self.action = DictCountWrapper(self.action)
            self.reward = DictCountWrapper(self.reward)

        # No done yet, use is_terminal?
        # self.done = self.parser.get_done(self.index)

    def get_unused_keys(self):
        unused_dict = {}
        for attr in ["goal", "state", "action", "reward"]:
            unused_dict[attr] = []
            for key in getattr(self, attr):
                if getattr(self, attr).access_counter[key] == 0:
                    unused_dict[attr].append(key)

        # print("UNUSED KEYS", unused_dict)
        return unused_dict

    def __repr__(self):
        return f"Timestep(index={self.index}, goal={self.goal}, state={self.state}, action={self.action}, reward={self.reward})"


class TimeStepDataset(torch.utils.data.Dataset):
    # TODO: Replace kwargs with config
    # @profile
    def __init__(
        self,
        dataset_path,
        parser,
        config,
        preprocessing_functions=None,
        filtering_functions=None,
        throw_error_if_not_enough_timesteps=True,
    ):
        self.dataset_path = dataset_path
        self.parser = parser
        self.integrate_rewards_to_go = config.integrate_rewards_to_go
        self.context_length = config.context_length
        self.future_horizon = config.future_horizon
        self.use_slots = config.get("use_slots", False)
        # print("use slots has been set to: ", self.use_slots)
        self.forecast_steps = config.get("forecast_steps", 1)
        self.action_gap = config.get("action_gap", 0)
        self.gen_masks_for_action = config.get("gen_masks_for_action", False)
        self.use_predicted_latent_with_gap = config.get(
            "use_predicted_latent_with_gap", False
        )
        if self.action_gap is None:
            self.action_gap = 0

        for attr in ["action", "state", "bev"]:
            attr_dict = config.get(attr, {})
            for s_attr in attr_dict:
                self.future_horizon = max(
                    self.future_horizon, attr_dict[s_attr].get("future_horizon", 0)
                )

        config.future_horizon = self.future_horizon

        # print(self.future_horizon)

        self.past_horizon = config.past_horizon
        # self._round_func = np.floor if config.drop_last else np.ceil
        self._round_func = np.ceil
        self.frame_stride = config.frame_stride
        self.inter_window_stride = config.inter_window_stride
        self.trim_first_and_last = config.trim_first_and_last
        self.trim_count = config.trim_count if config.trim_first_and_last else 0
        self.ignore_past_for_length = config.get("ignore_past_for_length", False)

        # print(self.parser.get_size())
        if self.trim_first_and_last:
            if self.parser.get_size() > self.trim_count * 2:
                self.num_steps = self.parser.get_size() - (2 * self.trim_count)
            else:
                self.num_steps = 0
        else:
            self.num_steps = self.parser.get_size()

        # print(self.num_steps)
        self.length = int(
            self._round_func(
                (
                    self.num_steps
                    - (
                        self.future_horizon
                        + (self.past_horizon if not self.ignore_past_for_length else 0)
                        + (self.action_gap if not self.ignore_past_for_length else 0)
                        + self.context_length
                    )
                    * self.frame_stride
                    - 1
                )
                / (config.inter_window_stride)
            )
        )
        # print(self.length)
        if self.length < 1:
            self.length = 0
        self.preprocessing_functions = preprocessing_functions
        self.filtering_functions = filtering_functions
        self.throw_error_if_not_enough_timesteps = throw_error_if_not_enough_timesteps
        self.use_future_ego_waypoints = config.use_future_ego_waypoints
        self.use_future_vehicle_forecast = config.use_future_vehicle_forecast
        self.use_past_horizon_states = config.get("use_past_horizon_states", False)

        #         print(f"""Dataset report:
        # path: {self.dataset_path}
        # self.num_steps: {self.num_steps}
        # self.length: {self.length}
        # self.future_horizon: {self.future_horizon}
        # self.past_horizon: {self.past_horizon}
        # self.context_length: {self.context_length}
        # ==TRIMMING==
        # self.trim_first_and_last: {self.trim_first_and_last}
        # self.trim_count: {self.trim_count}
        # starting index, assuming index 0: {self.trim_count}
        # starting index, assuming index 1: {self.inter_window_stride + self.trim_count}
        # indices retrieved assuming index 1: {[self.inter_window_stride + self.trim_count + i * self.frame_stride for i in range(self.context_length + self.future_horizon + self.past_horizon)]}
        # ===========
        # """)
        # exit(1)
        self.skip_keys_past = {}
        self.skip_keys_future = {}
        self.skip_keys_gap = {}
        self.past_keys_initialized = False

        self.create_goal_mask = config.get("create_goal_mask", False)

        self.metadata = {}

    def extract_metadata_from_path(self):
        # Make sure no trailing "/"
        # if
        # route_name = self.dataset_path.split("/")[-1]
        # route_name = self.dataset_path.split("/")
        pass

    def get_metadata_dict(self):
        pass

    def __len__(self):
        return self.length

    # @profile
    def __getitem__(self, idx):
        # Get timestep
        idx_start = idx * (self.inter_window_stride) + self.trim_count

        # print(self.context_length)

        # idx_end = min(idx_start + self.context_length + self.future_horizon, self.num_steps)
        num_instances = (
            self.context_length
            + self.future_horizon
            + self.past_horizon
            + self.action_gap
        )

        ending_idx = idx_start + (num_instances - 1) * self.frame_stride

        if self.ignore_past_for_length:
            ending_idx = (
                ending_idx - (self.past_horizon + self.action_gap) * self.frame_stride
            )

        if ending_idx > self.num_steps + self.trim_count * 2:
            if self.throw_error_if_not_enough_timesteps:
                import ipdb

                ipdb.set_trace()
                # raise IndexError(
                #     f"Index {idx_end} out of bounds for dataset of size {self.num_steps}"
                # )
            else:
                idx_end = self.num_steps

        # print(idx_start, idx_end, self.length)

        # timesteps = [
        #     Timestep(
        #         self.parser,
        #         idx_start + i * self.frame_stride,
        #         self.preprocessing_functions,
        #         self.filtering_functions,
        #     )
        #     for i in range(num_instances)
        # ]
        # print(idx, idx_start, ending_idx)
        timesteps = []
        for i in range(num_instances):
            if i < self.past_horizon:
                skip_keys = self.skip_keys_past
            elif i >= self.context_length + self.past_horizon + self.action_gap:
                skip_keys = self.skip_keys_future
            elif (
                i >= self.context_length + self.past_horizon
                and i < self.context_length + self.past_horizon + self.action_gap
            ) and not self.use_predicted_latent_with_gap:
                skip_keys = self.skip_keys_gap
            elif (
                i >= self.past_horizon and i < self.past_horizon + self.action_gap
            ) and self.use_predicted_latent_with_gap:
                skip_keys = self.skip_keys_gap
            else:
                skip_keys = {}

            tstep_idx = idx_start + i * self.frame_stride

            if self.ignore_past_for_length:
                tstep_idx = max(
                    tstep_idx
                    - (self.past_horizon + self.action_gap) * self.frame_stride,
                    self.trim_count,
                )

            # print("Timestep index", tstep_idx)

            timesteps.append(
                Timestep(
                    self.parser,
                    tstep_idx,
                    self.preprocessing_functions,
                    self.filtering_functions,
                    skip_keys_dict=skip_keys,
                )
            )

        if self.integrate_rewards_to_go:
            timesteps[-1].reward["rewards_to_go"] = 0
            for i in range(len(timesteps) - 2, -1, -1):
                timesteps[i].reward["rewards_to_go"] = (
                    timesteps[i + 1].reward["rewards_to_go"]
                    + timesteps[i + 1].reward["reward"]
                )

        # Extract useful state from past horizon, discard the rest
        if self.use_past_horizon_states:
            # raise NotImplementedError("Use past horizon states not implemented yet")
            assert (
                len(timesteps)
                == self.context_length
                + self.future_horizon
                + self.past_horizon
                + self.action_gap
            )
            timesteps[self.past_horizon].state["rgb_front"] = np.concatenate(
                [t.state["rgb_front"] for t in timesteps[: self.past_horizon + 1]],
                axis=0,
            )
        else:
            timesteps[self.past_horizon].state["rgb_front"] = np.concatenate(
                [timesteps[0].state["rgb_front"]],
                axis=0,
            )
            # print("State keys", timesteps[self.past_horizon].state.keys())
            # if "bevslotspercept" in timesteps[0].state:
            #     timesteps[self.past_horizon].state["bevslotspercept"] = np.concatenate(
            #         [
            #             t.state["bevslotspercept"]
            #             for t in timesteps[: self.past_horizon + 1]
            #         ],
            #         axis=0,
            #     )
            # # If past horizon is 0, unsqueeze for consistency
            # if self.past_horizon == 0:
            #     timesteps = np.expand_dims(timesteps, axis=0)

        past_time_steps = timesteps[: self.past_horizon]

        # Get rid of first past_horizon timesteps
        timesteps = timesteps[self.past_horizon :]

        if self.use_future_ego_waypoints:
            assert (
                len(timesteps)
                == self.context_length + self.future_horizon + self.action_gap
            ), (
                "Expected length "
                + str(self.context_length + self.future_horizon + self.action_gap)
                + " but got "
                + str(len(timesteps))
            )

            # waypoints = np.zeros(
            #     (self.context_length, self.future_horizon * 2)
            # )  # Each waypoint is 2D

            if not self.use_predicted_latent_with_gap:
                for i in range(self.context_length):
                    ego_states = [
                        t.action["ego_matrix"]
                        for t in timesteps[
                            i
                            + self.action_gap : i
                            + self.future_horizon
                            + 1
                            + self.action_gap
                        ]
                    ]
                    # Skip first one because it is the current ego matrix
                    transformed_ego_matrices = transform_waypoints(ego_states)[1:]

                    waypoints = np.stack(transformed_ego_matrices, axis=0)[:, :2, -1]

                    # waypoints = iterative_line_interpolation(waypoints, 10)

                    timesteps[i].action["waypoints"] = waypoints
            else:
                for i in range(self.context_length + self.action_gap):
                    ego_states = [
                        t.action["ego_matrix"]
                        for t in timesteps[i : i + self.future_horizon + 1]
                    ]
                    # Skip first one because it is the current ego matrix
                    # mn = np.mean(ego_states[0])

                    transformed_ego_matrices = transform_waypoints(ego_states)[1:]

                    waypoints = np.stack(transformed_ego_matrices, axis=0)[:, :2, -1]
                    # from .data_utils import interpolate_waypoints
                    # import ipdb; ipdb.set_trace();
                    # interpolate_waypoints(waypoints, 20)

                    # import ipdb; ipdb.set_trace()

                    # print(waypoints.shape)

                    # waypoints[0, 0] = mn

                    timesteps[i].action["waypoints"] = waypoints
        else:
            raise NotImplementedError(
                "Without future ego waypoints not implemented yet"
            )

        if self.use_future_vehicle_forecast:
            # raise NotImplementedError("Future vehicle forecast not implemented yet to account for action gap")
            extract_forecast_targets(
                timesteps,
                self.context_length + self.action_gap,
                self.future_horizon,
                use_slots=self.use_slots,
                forecast_steps=self.forecast_steps,
            )

        # Remove bevobjids, ego_matrix from state
        for i in range(self.context_length):
            timesteps[i].action.pop("ego_matrix", None)
            timesteps[i].state.pop("bevobjids", None)

        if self.use_predicted_latent_with_gap:
            if self.action_gap == 0:
                raise ValueError(
                    "Cannot use predicted latent with gap when action gap is 0"
                )

            for i in range(self.context_length, self.context_length + self.action_gap):
                timesteps[i].action.pop("ego_matrix", None)
                timesteps[i].state.pop("bevobjids", None)

            for i in range(self.context_length + self.action_gap, self.action_gap, -1):
                i = i - 1
                # import ipdb; ipdb.set_trace()
                timesteps[i].state["rgb_front"] = timesteps[i - self.action_gap].state[
                    "rgb_front"
                ]
                timesteps[i].state["target_rgb_front"] = timesteps[
                    i - self.action_gap
                ].state["target_rgb_front"]

                timesteps[i].state["frc_speed"] = timesteps[i - self.action_gap].state[
                    "speed"
                ]
                timesteps[i].state["frc_goal"] = timesteps[i - self.action_gap].goal[
                    "dual_target_point"
                ]
                timesteps[i].state["frc_wps"] = timesteps[i - self.action_gap].action[
                    "waypoints"
                ]

        if not self.past_keys_initialized:
            skip_keys_past_init = {}
            # print("Handling past")
            for i, t in enumerate(past_time_steps):
                if i == 0:
                    skip_keys_past_init = t.get_unused_keys()
                else:
                    cur_unused_keys = t.get_unused_keys()
                    for k in cur_unused_keys:
                        # Intersection of unused keys
                        skip_keys_past_init[k] = [
                            x for x in skip_keys_past_init[k] if x in cur_unused_keys[k]
                        ]
            self.skip_keys_past = skip_keys_past_init
            # print("Handling future")
            skip_keys_future_init = {}
            if self.future_horizon > 0:
                for i, t in enumerate(timesteps[-self.future_horizon :]):
                    if i == 0:
                        skip_keys_future_init = t.get_unused_keys()
                    else:
                        cur_unused_keys = t.get_unused_keys()
                        for k in cur_unused_keys:
                            # Intersection of unused keys
                            skip_keys_future_init[k] = [
                                x
                                for x in skip_keys_future_init[k]
                                if x in cur_unused_keys[k]
                            ]
            self.skip_keys_future = skip_keys_future_init

            skip_keys_action_gap_init = {}
            if self.action_gap > 0:
                if not self.use_predicted_latent_with_gap:
                    end_index = (
                        -self.future_horizon if self.future_horizon > 0 else None
                    )
                    start_idx = -self.future_horizon - self.action_gap
                else:
                    start_idx = 0
                    end_index = self.action_gap

                for i, t in enumerate(timesteps[start_idx:end_index]):
                    if i == 0:
                        skip_keys_action_gap_init = t.get_unused_keys()
                    else:
                        cur_unused_keys = t.get_unused_keys()
                        for k in cur_unused_keys:
                            # Intersection of unused keys
                            skip_keys_action_gap_init[k] = [
                                x
                                for x in skip_keys_action_gap_init[k]
                                if x in cur_unused_keys[k]
                            ]
            self.skip_keys_gap = skip_keys_action_gap_init
            self.past_keys_initialized = True
            # Print unused keys for each category
            # print("Past keys:", self.skip_keys_past)
            # print("Future keys:", self.skip_keys_future)
            # print("Action gap keys:", self.skip_keys_gap)

        if self.action_gap > 0 and self.use_predicted_latent_with_gap:
            timesteps = timesteps[self.action_gap :]

        if self.gen_masks_for_action:
            # all_masks = []
            for i in range(self.context_length):
                canvas = timesteps[0].state["rgb_front"][-1]

                # print(canvas.shape, timesteps[0].state["rgb_front"].shape)
                # print("Gen mask canvas shape")

                # import ipdb; ipdb.set_trace()

                if "waypoints" in timesteps[0].action:
                    waypoints_cur = (
                        point_to_canvas_coordinates_rel_to_center(
                            timesteps[0].action["waypoints"]
                            * np.asarray([1, -1]).reshape(-1, 2),
                            height=-1.6,
                        )
                        / 2
                    )
                if "path" in timesteps[0].action:
                    path_cur = (
                        point_to_canvas_coordinates_rel_to_center(
                            timesteps[0].action["path"], height=-1.6
                        )
                        / 2
                    )

                canvas = np.zeros_like(canvas)

                edge_size = canvas.shape[0]
                origin = (edge_size, edge_size // 2)

                canvas = draw_points_on_camera(
                    canvas,
                    waypoints_cur,
                    color=(0, 0, 255),
                    first_origin=origin,
                    radius=20,
                )
                canvas = draw_points_on_camera(
                    canvas, path_cur, color=(0, 255, 0), first_origin=origin, radius=10
                )

                # Convert to pillow image, save to /home/shadihamdan/viz/viz.png
                # from PIL import Image
                # Image.fromarray(canvas).save("/home/shadihamdan/viz/viz.png")

                timesteps[i].state["mask"] = canvas

        if self.create_goal_mask:
            if "rgb_front" in timesteps[0].state:
                for i in range(self.context_length):
                    if "target_point" in timesteps[i].goal:
                        target_points = timesteps[i].goal["target_point"]
                    else:
                        target_points = timesteps[i].goal["dual_target_point"]
                    target_points = (
                        point_to_canvas_coordinates_rel_to_center(
                            target_points.reshape(-1, 2), height=-1.6
                        )
                        / 2
                    )

                    canvas = timesteps[i].state["rgb_front"][-1]

                    canvas = np.asarray(canvas).copy()

                    edge_size = canvas.shape[0]
                    origin = (edge_size, edge_size // 2)

                    canvas = draw_points_on_camera(
                        canvas,
                        target_points,
                        color=(255, 0, 255),
                        first_origin=origin,
                        radius=10,
                        blur=8,
                    )

                    timesteps[i].state["goal_mask"] = canvas
                    # timesteps[i].state["rgb_front"] = canvas

        if self.future_horizon > 0:
            timesteps = timesteps[: -self.future_horizon]

        if self.action_gap > 0 and not self.use_predicted_latent_with_gap:
            timesteps = timesteps[: -self.action_gap]

        return timesteps

    def __getweight__(self, idx, reduce="last"):
        # idx_start = idx * self.context_length + self.trim_count

        # idx_end = min(idx_start + self.context_length, self.num_steps)
        # idx_start = idx * (self.inter_window_stride) + self.trim_count

        # # print(self.context_length)

        # # idx_end = min(idx_start + self.context_length + self.future_horizon, self.num_steps)
        # num_instances = (
        #     self.context_length
        #     + self.future_horizon
        #     + self.past_horizon
        #     + self.action_gap
        # )

        # ending_idx = idx_start + (num_instances - 1) * self.frame_stride
        idx_start = idx * (self.inter_window_stride) + self.trim_count

        # print(self.context_length)

        # idx_end = min(idx_start + self.context_length + self.future_horizon, self.num_steps)
        num_instances = (
            self.context_length
            + min(self.future_horizon, 4)
            + self.past_horizon
            + self.action_gap
        )

        ending_idx = idx_start + (num_instances - 1) * self.frame_stride

        if self.ignore_past_for_length:
            ending_idx = (
                ending_idx - (self.past_horizon + self.action_gap) * self.frame_stride
            )

        # weights = [
        #     self.parser.get_weight(idx * self.frame_stride)
        #     for idx in range(idx_start, idx_end)
        # ]
        weights = []
        for i in range(num_instances):
            if (
                i < self.past_horizon
                or i >= self.context_length + self.past_horizon + self.action_gap
            ):
                # print(i)
                continue
            # print(i)

            tstep_idx = idx_start + i * self.frame_stride

            if self.ignore_past_for_length:
                tstep_idx = max(
                    tstep_idx
                    - (self.past_horizon + self.action_gap) * self.frame_stride,
                    self.trim_count,
                )

            # print(idx_start, i, tstep_idx, idx_start + i * self.frame_stride)
            weights.append(self.parser.get_weight(tstep_idx))

        # print(weights)
        # import ipdb; ipdb.set_trace()

        if reduce == "mean":
            return np.mean(weights, axis=0)
        elif reduce == "sum":
            return np.sum(weights, axis=0)
        elif reduce == "last":
            return weights[-1]
        elif reduce == "first":
            return weights[0]
        elif reduce == "max":
            return np.max(weights, axis=0)
        else:
            raise NotImplementedError(f"Reduce method {reduce} not implemented")

    def __getnoisy__(self, idx, reduce="last"):
        idx_start = idx * self.context_length + self.trim_count

        idx_end = min(idx_start + self.context_length, self.num_steps)

        # To skip computing noisy for all timesteps, we can just return the last one if reduce is last
        if reduce == "last":
            return self.parser.get_noisy((idx_end - 1) * self.frame_stride)

        noisies = [
            self.parser.get_noisy(idx * self.frame_stride)
            for idx in range(idx_start, idx_end)
        ]

        if reduce == "any":
            return any(noisies)
        elif reduce == "all":
            return all(noisies)
        else:
            raise NotImplementedError(f"Reduce method {reduce} not implemented")

    def __repr__(self):
        return f"TimeStepDataset(dataset_path={self.dataset_path}, parser={self.parser}, integrate_rewards_to_go={self.integrate_rewards_to_go}, context_length={self.context_length}, drop_last={self._round_func == np.ceil}, frame_stride={self.frame_stride})"


class B2DSequenceDataset(torch.utils.data.Dataset):
    # @profile
    def __init__(
        self,
        dataset_path,
        split,
        config,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.config = config
        self.use_predicted_latent_with_gap = config.get(
            "use_predicted_latent_with_gap", False
        )
        self.gen_masks_for_action = config.get("gen_masks_for_action", False)
        routes = sorted(glob(join(dataset_path, "*")))

        # Shuffle with fixed seed 42 to make sure that the split is always the same
        rng = np.random.RandomState(42)
        rng.shuffle(routes)

        if split == "all":
            self.routes = routes
        else:
            # Split first 96% of routes into train, next 2% into val, and last 2% into test
            self.routes = (
                routes[: int(len(routes) * 0.94)]
                if split == "train"
                else (
                    routes[int(len(routes) * 0.94) : int(len(routes) * 0.97)]
                    if split == "val"
                    else routes[int(len(routes) * 0.97) :]
                )
            )

        filtering_functions = {}
        preprocessing_functions = {}

        if "rgb" in config.state_type:
            keys = [k for k in config.state_type.split("-") if "rgb" in k]
            for k in keys:
                prep = get_rgb_preprocessing_function_from_config(config)

                preprocessing_functions[k] = lambda x: np.asarray(prep(x))[
                    np.newaxis, ...
                ]

        # To avoid recomputing
        folder_to_ext = Parser.get_folder_to_ext(join(dataset_path, self.routes[0]))

        state_type = config.state_type
        action_type = config.action_type
        reward_type = config.reward_type
        goal_type = config.goal_type

        self.state_type = state_type
        self.action_type = action_type
        self.reward_type = reward_type
        self.goal_type = goal_type
        args = [
            (
                join(dataset_path, route),
                state_type,
                action_type,
                reward_type,
                goal_type,
                folder_to_ext,
            )
            for route in self.routes
        ]

        self.dataset_caching_cfg = config.dataset_caching
        if self.dataset_caching_cfg.enabled and self.dataset_caching_cfg.cache_metadata:
            size_cache_path = join(
                self.dataset_caching_cfg.cache_dir, "size_cache.json"
            )
            size_dict = {}

            if os.path.exists(size_cache_path):
                try:
                    size_dict = json.load(open(size_cache_path, "r"))
                except:
                    size_dict = {}

            args = [tuple([*arg, size_dict.get(arg[0], None)]) for arg in args]

            all_cached = all([arg[-1] is not None for arg in args])
        else:
            args = [tuple([*arg, None]) for arg in args]

        if config.parallel_dataset_init:
            # use pool of 16 workers to create it in the same order
            from multiprocessing import Pool

            with Pool(config.parallel_dataset_workers) as p:
                self.route_data_parsers = p.map(create_parse_from_route, args)
        else:
            self.route_data_parsers = [create_parse_from_route(arg) for arg in args]

        if self.dataset_caching_cfg.enabled:
            if self.dataset_caching_cfg.cache_metadata:
                if not all_cached:
                    if os.path.exists(size_cache_path):
                        try:
                            size_dict = json.load(open(size_cache_path, "r"))
                        except:
                            size_dict = {}
                    else:
                        size_dict = {}

                    for prs in self.route_data_parsers:
                        size_dict[prs.root_path] = prs.get_size()

                    with open(size_cache_path, "w") as f:
                        json.dump(size_dict, f, indent=4)

            if self.dataset_caching_cfg.cache_slow_attributes:
                path_cache_dir = join(self.dataset_caching_cfg.cache_dir, "path_cache")
                os.makedirs(path_cache_dir, exist_ok=True)
                if config.parallel_dataset_init:
                    with Pool(config.parallel_dataset_workers) as p:
                        p.map(
                            cache_path_of_parser,
                            [(prs, path_cache_dir) for prs in self.route_data_parsers],
                        )
                else:
                    for prs in self.route_data_parsers:
                        prs.cache_path(path_cache_dir)

                for prs in self.route_data_parsers:
                    prs.load_path_cache(path_cache_dir)

                buckets_cache_dir = join(
                    self.dataset_caching_cfg.cache_dir, "buckets_cache"
                )
                os.makedirs(buckets_cache_dir, exist_ok=True)
                if config.parallel_dataset_init:
                    with Pool(config.parallel_dataset_workers) as p:
                        p.map(
                            cache_buckets_of_parser,
                            [
                                (prs, buckets_cache_dir)
                                for prs in self.route_data_parsers
                            ],
                        )
                else:
                    for prs in self.route_data_parsers:
                        prs.cache_buckets(buckets_cache_dir)

                for prs in self.route_data_parsers:
                    prs.load_bucket_cache(buckets_cache_dir)

        self.route_datasets = [
            TimeStepDataset(
                join(dataset_path, route),
                parser,
                config,
                preprocessing_functions=preprocessing_functions,
                filtering_functions=filtering_functions,
            )
            for route, parser in zip(self.routes, self.route_data_parsers)
        ]

        self.routes_num_sequences = [len(dataset) for dataset in self.route_datasets]

        self.total_num_sequences = np.sum(self.routes_num_sequences)
        if config.max_instances and config.max_instances > 0:
            self.total_num_sequences = min(
                self.total_num_sequences, config.max_instances
            )

        self.cumsum_num_sequences = np.cumsum(self.routes_num_sequences)

        self.frame_stride = config.frame_stride

        if config.skip_noisy:
            # Iterate over all routes and remove noisy sequences if last element is noisy
            self.valid_route_idxes = []
            noisy_list = self.getnoisy()
            for i, n in enumerate(noisy_list):
                if not n:
                    self.valid_route_idxes.append(i)
            print(
                "Skipped all noisy sequences. Num noisy sequences:",
                len(noisy_list) - len(self.valid_route_idxes),
                "out of",
                len(noisy_list),
                "sequences.",
            )
            self.total_num_sequences = (
                len(self.valid_route_idxes)
                if config.max_instances < 0
                else min(len(self.valid_route_idxes), config.max_instances)
            )
        else:
            self.valid_route_idxes = None

        self.integrate_rewards_to_go = config.integrate_rewards_to_go

    def __len__(self):
        return self.total_num_sequences

    # @profile
    def __getitem__(self, idx):
        try:
            if self.valid_route_idxes is not None:
                idx = self.valid_route_idxes[idx]
        except:
            import ipdb

            ipdb.set_trace()

        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        sequence = self.route_datasets[route_idx][idx]

        # Flatten and convert to tensors
        sequence = self.flatten_time_sequence(sequence)

        return sequence

    # Get the weight of the sequence based on a custom heuristic
    # Similar to getitem, but calls the internal getweight function
    def __getweight__(self, idx):
        if self.valid_route_idxes is not None:
            idx = self.valid_route_idxes[idx]
        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        weight = self.route_datasets[route_idx].__getweight__(idx)

        return weight

    # Get whether a sequence is noisy or not
    # Similar to getitem, but calls the internal getweight function
    def __getnoisy__(self, idx):
        # Binary search to find the correct route
        route_idx = np.searchsorted(self.cumsum_num_sequences, idx, side="right")

        idx -= self.cumsum_num_sequences[route_idx - 1] if route_idx > 0 else 0

        # Get timestep sequence
        noisy = self.route_datasets[route_idx].__getnoisy__(idx)

        return noisy

    def flatten_time_sequence(
        self, sequence, attributes=["goal", "state", "action", "reward"]
    ):
        assert len(sequence) != 0
        result = {}
        for attrib in attributes:
            if attrib == "state":
                continue
            result[attrib] = self.flatten_dicts(
                [getattr(timestep, attrib) for timestep in sequence]
            )

        if "state" in attributes:
            state_dict = self.flatten_state_dict(
                [getattr(timestep, "state") for timestep in sequence]
            )
            for k in state_dict:
                result[k] = state_dict[k]

        return result

    def flatten_dicts(self, dicts):
        flattened = {k: [] for k in dicts[0]}

        for dict in dicts:
            for k in flattened:
                flattened[k].append(dict[k])

        for k in flattened:
            if isinstance(flattened[k][0], np.ndarray):  # Array
                dtype = (
                    np.float32
                    if np.issubdtype(flattened[k][0].dtype, np.floating)
                    else np.int64
                )
                flattened[k] = np.stack(flattened[k]).astype(dtype)
            else:  # Scalar
                dtype = np.float32 if type(flattened[k][0]) is float else np.int64
                flattened[k] = np.asarray(flattened[k], dtype=dtype)
                flattened[k] = np.expand_dims(flattened[k], axis=-1)

        keys = [k for k in dicts[0]]

        keys = self.sort_keys(keys)

        flattened_tensor = torch.tensor(
            np.concatenate([flattened[k] for k in keys], axis=1)
        )

        return flattened_tensor

    def sort_keys(self, keys):
        # Sort keys so that the order is always the same
        # This is important to make sure that the model always gets the same input
        return sorted(keys)

    # We would like to treat rgb differently
    def flatten_state_dict(self, dicts):
        flattened = {k: [] for k in dicts[0]}

        for dict in dicts:
            for k in flattened:
                flattened[k].append(dict[k])

        sorted_keys = sorted(k for k in dicts[0])

        result = {}
        flattened_tensor = torch.tensor(
            np.stack(
                [
                    flattened[k]
                    for k in sorted_keys
                    if not ("rgb" in k or "mask" in k or "frc" in k)
                ]
            ).astype(np.float32)
        ).T

        result["state"] = flattened_tensor

        # concatenate and add every rgb state separately to the result
        for k in sorted_keys:
            if "rgb" in k:
                # Store image in 8bit uint
                flattened_tensor = torch.tensor(
                    np.concatenate(flattened[k]).astype(np.uint8)
                )
                result[k] = flattened_tensor
            if "mask" in k:
                flattened_tensor = torch.tensor(np.stack(flattened[k]).astype(np.uint8))
                result[k] = flattened_tensor
            if "frc" in k:
                flattened_tensor = torch.tensor(
                    np.stack(flattened[k]).astype(np.float32)
                )
                result[k] = flattened_tensor

        return result

    # Get the weights of all sequences. This function caches the weights to avoid recomputing them. The cache is stored on disk in the
    # .cache folder, and the name of the file is computed from the hash of the state variables as well as the hash of the
    # length of the dataset. This means that if the dataset is modified, the cache will be invalidated.
    def getweights(self):
        # Check if the cache exists
        cache_path = join(os.path.dirname(os.path.realpath(__file__)), ".cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # Calculate the hash of the state variables
        state_hash = hashlib.md5(
            f"{self.dataset_path}-{self.routes_num_sequences}-{self.total_num_sequences}-{self.frame_stride}".encode(
                "utf-8"
            )
        ).hexdigest()

        # Calculate the hash of the length of the dataset
        length_hash = hashlib.md5(
            f"{self.total_num_sequences}".encode("utf-8")
        ).hexdigest()

        instance_sample_weights_hash = hashlib.md5()

        for i in range(0, self.total_num_sequences, 100):
            instance_sample_weights_hash.update(
                f"{self.__getweight__(i)}".encode("utf-8")
            )

        instance_sample_weights_hash = instance_sample_weights_hash.hexdigest()

        # Check if the cache exists
        cache_file = join(
            cache_path,
            f"{state_hash}-{length_hash}-{instance_sample_weights_hash}-weights.npy",
        )

        if os.path.exists(cache_file):
            print("Loading weights from cache")
            return np.load(cache_file)

        # If the cache does not exist, compute the weights and save them to the cache
        weights = []
        from tqdm import trange

        for i in trange(self.total_num_sequences):
            weights.append(self.__getweight__(i))

        np.save(cache_file, np.asarray(weights))

        return weights

    def get_bucket_names(self):
        if len(self.route_data_parsers) == 0:
            return []

        if not hasattr(self.route_data_parsers[0], "bucket_names"):
            return []

        return self.route_data_parsers[0].bucket_names

    def getnoisy(self):
        cache_path = join(os.path.dirname(os.path.realpath(__file__)), ".cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # Calculate the hash of the state variables
        state_hash = hashlib.md5(
            f"{self.dataset_path}-{self.routes_num_sequences}-{self.total_num_sequences}-{self.frame_stride}".encode(
                "utf-8"
            )
        ).hexdigest()

        # Calculate the hash of the length of the dataset
        length_hash = hashlib.md5(
            f"{self.total_num_sequences}".encode("utf-8")
        ).hexdigest()

        # Check if the cache exists
        cache_file = join(cache_path, f"{state_hash}-{length_hash}-noisy.npy")

        if os.path.exists(cache_file):
            print("Loading noise weights from cache")
            return np.load(cache_file)

        # If the cache does not exist, check whether every instance is noisy or not and and save them to the cache
        noisy = []

        # for i in trange(self.total_num_sequences, desc="Skipping noisy sequences"):
        # noisy.append(self.__getnoisy__(i))
        noisy = [False] * self.total_num_sequences

        np.save(cache_file, np.asarray(noisy))

        return noisy

    def get_parametrized_dirname(self):
        dataset_path_name = os.path.basename(self.dataset_path)

        attributes = [
            "use_future_ego_waypoints",
            "future_horizon",
            "past_horizon",
            "trim_first_and_last",
            "trim_count",
            "forecast_steps",
            "action_type",
            "state_type",
            "goal_type",
            "reward_type",
            "bev_type",
            "context_length",
            "frame_stride",
            "gen_masks_for_action",
            "create_goal_mask",
            "use_future_vehicle_forecast",
            "use_predicted_latent_with_gap",
        ]

        appended_attributes = [
            "action_gap",
        ]  # Only added if not None

        subattributes = [
            "rgb_crop.type",
            "rgb_crop.crop_size",
            "rgb_crop.resize",
        ]

        attr_string = ""
        for attr in attributes:
            attr_string += f"{attr}={getattr(self.config, attr, None)}"

        for attr in appended_attributes:
            attr_val = getattr(self.config, attr, None)
            if attr_val is not None:
                attr_string += f"{attr}={attr_val}"

        for attr in subattributes:
            subattrs = attr.split(".")
            attr_val = self.config
            for subattr in subattrs:
                attr_val = getattr(attr_val, subattr, None)
                if attr_val is None:
                    break
            if attr_val is not None:
                attr_string += f"{attr}={attr_val}"

        state_hash = hashlib.md5(attr_string.encode("utf-8")).hexdigest()

        return "{}_{}_{}_{}".format(
            dataset_path_name, state_hash[:10], self.split, self.total_num_sequences
        )


def create_parse_from_route(args):
    pth, state_type, action_type, reward_type, goal_type, folder_to_ext, size = args
    return Parser(
        pth,
        state_type,
        action_type,
        reward_type,
        goal_type,
        folder_to_ext=folder_to_ext,
        size=size,
    )


def cache_path_of_parser(parser, cache_dir=None):
    if cache_dir is None:
        parser, cache_dir = parser
    parser.cache_path(cache_dir)


def cache_buckets_of_parser(parser, cache_dir=None):
    if cache_dir is None:
        parser, cache_dir = parser
    parser.cache_buckets(cache_dir)
