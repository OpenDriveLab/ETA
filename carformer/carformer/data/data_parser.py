# import orjson as json
import json
import gzip
import os
from PIL import Image
import numpy as np
from .data_utils import (
    iterative_line_interpolation,
    get_hazard_directions,
    is_walker_hazard,
)

home_dir = os.environ["HOME"]
path_cache = {}


class Parser:
    def __init__(
        self,
        root_path,
        state_type,
        action_type,
        reward_type,
        goal_type,
        folder_to_ext=None,
        size=None,
        cache_dir=None,
    ):
        self.root_path = root_path
        self.state_type = state_type
        self.action_type = action_type
        self.reward_type = reward_type
        self.goal_type = goal_type

        self.states = state_type.split("-")
        self.actions = action_type.split("-")
        self.rewards = reward_type.split("-")
        self.goals = goal_type.split("-")

        if folder_to_ext is None:
            self._folder_to_ext = {}
            # Check the extension of the first file in every folder in path
            for folder in os.listdir(self.root_path):
                folder_path = os.path.join(self.root_path, folder)
                if os.path.isdir(folder_path):
                    fl = os.listdir(folder_path)
                    # Get extension
                    if len(fl) > 0:
                        ext = os.path.splitext(fl[0])[1]
                        self._folder_to_ext[folder] = ext
        else:
            self._folder_to_ext = folder_to_ext

        if size is not None:
            self.length = size
        else:
            # check if self.root_path/anno/00000.json.gz exists. If not, set length to 0 and return
            sanity_file = os.path.join(self.root_path, "anno", "00000.json.gz")
            if not os.path.exists(sanity_file):
                self.length = 0
                return
            self.length = len(os.listdir(os.path.join(self.root_path, "anno")))

        self.cache_dir = cache_dir
        self.path_cache = None

    def get_state(
        self,
        idx,
        preprocessing_functions=None,
        filtering_functions=None,
        skip_keys=None,
    ):
        ts_prefix = str(idx).zfill(5)
        state_dict = self.gzip_json_load(os.path.join("anno", f"{ts_prefix}.json.gz"))

        state = {}

        for s in self.states:
            if skip_keys is not None and s in skip_keys:
                continue
            if "rgb" in s:
                rgb = Image.open(
                    os.path.join(self.root_path, "camera", s, f"{ts_prefix}.jpg")
                )

                speed = state_dict["speed"]

                action = np.mean(state_dict["bounding_boxes"][0]["world2ego"])

                state[s] = rgb
            elif s == "speed":
                state["speed"] = state_dict["speed"]
            else:
                raise ValueError(f"State type {s} not recognized")

            if preprocessing_functions is not None and s in preprocessing_functions:
                state[s] = preprocessing_functions[s](state[s])

        return state

    def get_action(self, idx, include_noise=False, skip_keys=None):
        ts_prefix = str(idx).zfill(5)
        state_dict = self.gzip_json_load(os.path.join("anno", f"{ts_prefix}.json.gz"))

        action = {}

        for a in self.actions:
            if skip_keys is not None and a in skip_keys:
                continue
            if a == "waypoints":
                # Get the current ego matrix from the measurement dict
                if skip_keys is not None and "ego_matrix" in skip_keys:
                    continue
                action["ego_matrix"] = state_dict["bounding_boxes"][0]["world2ego"]
            elif a == "path":
                # Default to 20 waypoints like CarLLaVA
                ego_matrix = state_dict["bounding_boxes"][0]["world2ego"]
                pointer_idx = idx
                points = []
                if self.path_cache is not None:
                    relevant_points = self.path_cache[pointer_idx:]

                    relevant_points_inv = np.linalg.inv(relevant_points)
                    ego_matrix = np.asarray(ego_matrix)
                    points = np.einsum(
                        "Xi, BiY -> BXY", ego_matrix, relevant_points_inv
                    )[:, :2, 3]

                    # Multiply by -1 to get the correct y coordinate
                    points[:, 1] = -points[:, 1]
                else:
                    while True:
                        print("Rebuilding path cache")
                        path = os.path.join(
                            self.root_path,
                            "anno",
                            f"{str(pointer_idx).zfill(5)}.json.gz",
                        )

                        if path in path_cache:
                            cur_point = path_cache[path]
                        else:
                            if not os.path.exists(path):
                                break

                            cur_dct = self.gzip_json_load(
                                os.path.join(
                                    "anno", f"{str(pointer_idx).zfill(5)}.json.gz"
                                )
                            )
                            cur_point = cur_dct["bounding_boxes"][0]["world2ego"]

                        x, y = np.dot(ego_matrix, np.linalg.inv(cur_point))[:2, 3]

                        points.append((x, -y))
                        pointer_idx += 1

                path = iterative_line_interpolation(points)

                action["path"] = path
            else:
                raise ValueError(f"Action type {a} not recognized")

        if include_noise:
            return action, False
        else:
            return action

    def get_noisy(self, idx):
        return False

    def get_reward(self, idx, skip_keys=None):
        rewards = {}

        for r in self.rewards:
            if skip_keys is not None and r in skip_keys:
                continue
            rewards[r] = 0.0

        return rewards

    def get_goal(self, idx, skip_keys=None):
        ts_prefix = str(idx).zfill(5)
        state_dict = self.gzip_json_load(os.path.join("anno", f"{ts_prefix}.json.gz"))

        goal = {}

        for g in self.goals:
            if skip_keys is not None and g in skip_keys:
                continue
            if g == "target_point":
                # Get the target point from the state
                ego = state_dict["bounding_boxes"][0]
                theta = ego["rotation"][-1] * np.pi / 180

                command_near_xy = np.array(
                    [
                        state_dict["x_command_near"] - state_dict["x"],
                        -state_dict["y_command_near"] + state_dict["y"],
                    ]
                )
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                )
                local_command_xy = rotation_matrix @ command_near_xy
                # command_far_xy = np.array([state_dict["x_command_far"]-state_dict['x'],-state_dict["y_command_far"]+state_dict['y']])
                # local_command_far_xy = rotation_matrix @ command_far_xy

                goal["target_point"] = local_command_xy
            elif g == "dual_target_point":
                # Get the target point from the state
                # local_command_point = np.array(state_dict["target_point"])
                # goal["target_point"] = local_command_point
                ego = state_dict["bounding_boxes"][0]
                theta = ego["rotation"][-1] * np.pi / 180

                command_near_xy = np.array(
                    [
                        state_dict["x_command_near"] - state_dict["x"],
                        -state_dict["y_command_near"] + state_dict["y"],
                    ]
                )
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                )
                local_command_xy = rotation_matrix @ command_near_xy
                command_far_xy = np.array(
                    [
                        state_dict["x_command_far"] - state_dict["x"],
                        -state_dict["y_command_far"] + state_dict["y"],
                    ]
                )
                local_command_far_xy = rotation_matrix @ command_far_xy

                goal["dual_target_point"] = np.stack(
                    [local_command_xy, local_command_far_xy], axis=0
                )
            else:
                raise ValueError(f"Goal type {g} not recognized")

        return goal

    def get_size(self):
        return self.length

    def get_weight(self, idx):
        return self.get_buckets(idx)

    def get_buckets(self, idx):
        # Get which bucket the current idx belongs to
        assert self.buckets is not None, "Buckets not initialized"

        return self.buckets[idx]

    @staticmethod
    def get_folder_to_ext(dir):
        folder_to_ext = {}
        for folder in os.listdir(dir):
            folder_path = os.path.join(dir, folder)
            if os.path.isdir(folder_path):
                fl = os.listdir(folder_path)
                # Get extension
                if len(fl) > 0:
                    ext = os.path.splitext(fl[0])[1]
                    folder_to_ext[folder] = ext
                else:
                    raise ValueError(f"Folder {folder} is empty in directory {dir}")

        return folder_to_ext

    def gzip_json_load(self, rel_file_path, root_path=None):
        if root_path is None:
            root_path = self.root_path
        with gzip.open(os.path.join(root_path, rel_file_path), "r") as f:
            return json.loads(f.read().decode("utf-8"))

    def path_is_cached(self, cache_dir):
        return os.path.exists(
            os.path.join(cache_dir, f"{self.root_path.split('/')[-1]}.json.gz")
        )

    def bucket_is_cached(self, path_cache_dir):
        return os.path.exists(
            os.path.join(path_cache_dir, f"{self.root_path.split('/')[-1]}.npz")
        )

    def cache_buckets(self, bucket_cache_dir):
        if self.bucket_is_cached(bucket_cache_dir):
            return

        bucket_names, all_buckets = self.get_all_buckets()

        cache_path = os.path.join(
            bucket_cache_dir, f"{self.root_path.split('/')[-1]}.npz"
        )

        np.savez(cache_path, bucket_names=bucket_names, buckets=all_buckets)

    def cache_path(self, cache_dir):
        if self.path_is_cached(cache_dir):
            self.load_path_cache(cache_dir)
            return

        if "path" in self.action_type:
            all_paths = self.get_all_paths()

            cache_path = os.path.join(
                cache_dir, f"{self.root_path.split('/')[-1]}.json.gz"
            )

            with gzip.open(cache_path, "w") as f:
                f.write(json.dumps(all_paths).encode("utf-8"))

    def get_all_paths(self):
        if "path" not in self.action_type:
            return None
        all_egos = []
        for i in range(self.length):
            ts_prefix = str(i).zfill(5)
            state_dict = self.gzip_json_load(
                os.path.join("anno", f"{ts_prefix}.json.gz")
            )
            cur_point = state_dict["bounding_boxes"][0]["world2ego"]
            all_egos.append(cur_point)
        return all_egos

    def get_all_buckets(self):
        # Buckets are useful for bucketed sampling
        # Buckets defined as follows:
        # acceleration buckets:
        idx = 0
        all_state_dicts = []
        from skit.profiling import Ticker

        t = Ticker(verbose=False, track=True)

        for i in range(self.length):
            ts_prefix = str(i).zfill(5)
            state_dict = self.gzip_json_load(
                os.path.join("anno", f"{ts_prefix}.json.gz")
            )
            all_state_dicts.append(state_dict)

        swerving_scenarios = [
            "Accident",
            "BlockedIntersection",
            "ConstructionObstacle",
            "HazardAtSideLane",
            "ParkedObstacle",
            "VehicleOpensDoorTwoWays",
        ]

        data_is_swerving = any([x in self.root_path for x in swerving_scenarios])

        all_buckets = []
        general_bucket_name = ["general"]
        acceleration_bucket_names = [
            "acc_scratch",
            "acc_light_pedal",
            "acc_medium_pedal",
            "acc_heavy_pedal",
            "acc_brake",
            "acc_coast",
        ]
        steer_bucket_names = [
            "steer_right",
            "steer_left",
        ]
        vehicle_hazard_bucket_names = [
            "vehicle_hazard_front",
            "vehicle_hazard_back",
            "vehicle_hazard_side",
        ]
        stop_sign_bucket_names = ["stop_sign"]
        red_light_bucket_names = ["red_light"]
        swerving_bucket_names = ["swerving"]
        pedestrian_bucket_names = ["pedestrian"]

        for i in range(self.length):
            acceleration = all_state_dicts[i]["throttle"]
            brake = all_state_dicts[i]["brake"]
            speed = all_state_dicts[i]["speed"]

            acceleration_bucket = [
                1 if (acceleration > 0.2 and brake < 1.0 and speed < 0.05) else 0,
                1 if (acceleration > 0.2 and acceleration < 0.5) else 0,
                1 if (acceleration > 0.5 and acceleration < 0.9) else 0,
                1 if (acceleration > 0.9) else 0,
                1 if (brake > 0.2) else 0,
                1 if (acceleration < 0.2 and brake < 1.0) else 0,
            ]

            steer = all_state_dicts[i]["steer"]

            steer_bucket = [
                1 if (steer > 0.2) else 0,
                1 if (steer < -0.2) else 0,
            ]

            # other_vehicles = all_state_dicts[i]["vehicle_hazard"]

            vehicle_hazard_bucket = get_hazard_directions(
                all_state_dicts[i]["bounding_boxes"]
            )
            # print(i, vehicle_hazard_bucket)

            vehicle_hazard_bucket = [
                (
                    1 if any([x < 30 for x in vehicle_hazard_bucket]) else 0
                ),  # Heading from front
                (
                    1 if any([x > 150 for x in vehicle_hazard_bucket]) else 0
                ),  # Heading from back
                (
                    1 if any([x > 30 and x < 150 for x in vehicle_hazard_bucket]) else 0
                ),  # Heading from side
            ]

            if data_is_swerving and abs(steer) > 0.1:
                swerving_bucket = 1
            else:
                swerving_bucket = 0

            all_objs = all_state_dicts[i]["bounding_boxes"]

            stopsigns = [
                x
                for x in all_objs
                if x["class"] == "traffic_sign" and x["type_id"] == "traffic.stop"
            ]

            stop_sign_bucket = 1 if any([x["affects_ego"] for x in stopsigns]) else 0

            redtrafficlights = [
                x for x in all_objs if x["class"] == "traffic_light" and x["state"] == 0
            ]

            red_light_bucket = (
                1 if any([x["affects_ego"] for x in redtrafficlights]) else 0
            )

            pedestrian_hazards = is_walker_hazard(all_objs)
            pedestrian_hazard_bucket = 1 if pedestrian_hazards else 0

            instance_buckets = (
                [1]
                + acceleration_bucket
                + steer_bucket
                + vehicle_hazard_bucket
                + [
                    stop_sign_bucket,
                    red_light_bucket,
                    swerving_bucket,
                    pedestrian_hazard_bucket,
                ]
            )

            all_buckets.append(instance_buckets)

        bucket_names = (
            general_bucket_name
            + acceleration_bucket_names
            + steer_bucket_names
            + vehicle_hazard_bucket_names
            + stop_sign_bucket_names
            + red_light_bucket_names
            + swerving_bucket_names
            + pedestrian_bucket_names
        )

        all_buckets = np.asarray(all_buckets)

        return bucket_names, all_buckets

    def validate_cache(self, path_cache_dir):
        if not "path" in self.action_type:
            return True

        to_compare = self.load_path_cache(path_cache_dir)

        all_egos = self.get_all_paths()

        return np.allclose(np.asarray(all_egos), np.asarray(to_compare))

    def load_path_cache(self, path_cache_dir):
        if not "path" in self.action_type:
            return None

        cache_path = os.path.join(
            path_cache_dir, f"{self.root_path.split('/')[-1]}.json.gz"
        )
        if not os.path.exists(cache_path):
            return None

        path_cache = self.gzip_json_load(cache_path, root_path="")

        self.path_cache = np.asarray(path_cache)

    def load_bucket_cache(self, bucket_cache_dir):
        cache_path = os.path.join(
            bucket_cache_dir, f"{self.root_path.split('/')[-1]}.npz"
        )
        if not os.path.exists(cache_path):
            return None

        bucket_cache = np.load(cache_path)

        bucket_names = bucket_cache["bucket_names"]
        # cast names to string
        bucket_names = [str(x) for x in bucket_names]
        buckets = bucket_cache["buckets"]

        self.buckets = buckets
        self.bucket_names = bucket_names
