import os
import json
import math
from omegaconf import OmegaConf
from PIL import Image

import cv2
import torch
import numpy as np
import carla

from Bench2DriveZoo.team_code.planner import RoutePlanner

from scipy.optimize import fsolve

from carformer.ponderer import Ponderer
from carformer.visualization.visutils import (
    visualize_trajectory_action_predictions,
)

from carformer.config.carformer_config import CarformerConfig
from carformer.data import get_rgb_preprocessing_function

from leaderboard.autoagents import autonomous_agent

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

import scipy


def get_entry_point():
    return "PlanTAgent"


def sqrt_func(x):
    try:
        result = scipy.linalg.cholesky(x)
    except scipy.linalg.LinAlgError:
        x = (x + x.T) / 2
        result = scipy.linalg.cholesky(x)
    return result


class PlanTAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS

        if isinstance(path_to_conf_file, str):
            raise ValueError("Path to conf file should be a list")
            model_dir = path_to_conf_file.split("+")[0]
            self.config = CarformerConfig.from_pretrained(
                model_dir,
            )
        else:
            self.cfg = path_to_conf_file
            model_dir = self.cfg.agent_root
            self.config = CarformerConfig.from_pretrained(
                model_dir,
            )

        self.sim_fps = 20

        # Filtering
        self.points = MerweScaledSigmaPoints(
            # n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x, sqrt_method=sqrt_func#, sqrt_method=lambda x: np.zeros_like(x)
            n=4,
            alpha=0.0001,
            beta=2,
            kappa=0,
            subtract=residual_state_x,
            sqrt_method=sqrt_func,  # , sqrt_method=lambda x: np.zeros_like(x)
        )

        self.all_points = [[] for i in range(10)]

        import datetime

        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        self.trajectory_viz_dir = os.path.join(
            os.path.expanduser("~"), "viz/trajectories_viz/", time
        )
        os.makedirs(self.trajectory_viz_dir, exist_ok=True)
        self.ukf_logs_dir = os.path.join(os.path.expanduser("~"), "viz/ukf_logs/")
        os.makedirs(self.ukf_logs_dir, exist_ok=True)
        self.ukf_logs_dir = os.path.join(self.ukf_logs_dir, f"{time}.jsonl")

        self.ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=bicycle_model_forward,
            hx=measurement_function_hx,
            dt=1 / self.sim_fps,
            points=self.points,
            x_mean_fn=state_mean,
            z_mean_fn=measurement_mean,
            residual_x=residual_state_x,
            residual_z=residual_measurement_h,
        )

        # # State noise, same as measurement because we
        # # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise

        # # Used to set the filter state equal the first measurement
        self.filter_initialized = False

        self.ignore_filter = False

        self.step = 0
        self.initialized = False
        self.steer_damping = self.cfg.steer_damping

        model_name = model_dir.split("/")[-1]
        self.model = Ponderer.from_pretrained(
            model_dir,
            epoch=self.cfg.epoch_num,
            deepspeed="auto",
        ).eval()
        self.model_name = model_name

        if self.cfg.use_gt_frc is not None:
            self.model.cfg.training["use_gt_frc"] = self.cfg.use_gt_frc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(self.config)

        self.rgb_prep = get_rgb_preprocessing_function(
            self.config.training["rgb_crop"]["type"],
            self.config.training["rgb_crop"]["crop_size"],
            self.config.training["rgb_crop"]["resize"],
        )

        self.context = []
        self.data_fps = self.config.dataset["fps"]
        self.frame_stride = self.config.training["frame_stride"]
        self.context_stride = int(self.sim_fps / self.data_fps) * self.frame_stride
        self.context_length_training = self.config.training["context_length"]
        self.past_horizon_training = self.config.training["past_horizon"]

        self.action_gap = self.config.training.get("action_gap", 0)

        if self.action_gap is None:
            self.action_gap = 0

        # self.action_gap = 2

        self.use_predicted_latent_with_gap = self.config.training.get(
            "use_predicted_latent_with_gap", False
        )
        self.utilize_fast_current_latent = self.config.training.get(
            "utilize_fast_current_latent", False
        )

        self.delay_action_frames = int(
            self.action_gap
            * self.config.training["frame_stride"]
            * self.sim_fps
            / self.config.dataset["fps"]
        )
        # print(self.delay_action_frames)

        self.control_preds = []

        if not self.use_predicted_latent_with_gap:
            for i in range(self.delay_action_frames):
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                self.control_preds.append(control)

        self.total_sequence_length = (
            self.context_length_training + self.past_horizon_training
        )

        if self.use_predicted_latent_with_gap:
            self.context_array_max_length = (
                1
                + (self.total_sequence_length - 1) * self.context_stride
                + self.delay_action_frames
            )
        else:
            self.context_array_max_length = (
                1 + (self.total_sequence_length - 1) * self.context_stride
            )

        self.scenario_logger = False

        self.turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=1.75, K_I=1.0, K_D=2.0, n=20)
        print(os.getcwd())

        self.stuck_detector = 0
        self.forced_move = False
        self.forced_move_counter = 0
        self.infraction_count = 0

        with open(os.path.join("evalconfig.yml"), "w") as f:
            OmegaConf.save(vars(self.cfg), f)

        main_ckpt_dir = self.cfg.checkpoint
        self.metrics_path = main_ckpt_dir.replace("/evallogs/", "/metriclogs/")
        metrics_dir = os.path.dirname(self.metrics_path)

        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        self.config.save_pretrained("evalmodelconfig")

    def _init(self):
        try:
            locx, locy = (
                self._global_plan_world_coord[0][0].location.x,
                self._global_plan_world_coord[0][0].location.y,
            )
            lon, lat = self._global_plan[0][0]["lon"], self._global_plan[0][0]["lat"]
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = (
                    lon * math.cos(x * math.pi / 180)
                    - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)
                    - math.cos(x * math.pi / 180) * y
                )
                eq2 = (
                    math.log(math.tan((lat + 90) * math.pi / 360))
                    * EARTH_RADIUS_EQUA
                    * math.cos(x * math.pi / 180)
                    + locy
                    - math.cos(x * math.pi / 180)
                    * EARTH_RADIUS_EQUA
                    * math.log(math.tan((90 + x) * math.pi / 360))
                )
                return [eq1, eq2]

            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors = []
        sensors += [
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "CAM_FRONT",
            },
            # imu
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            # gps
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
            # speed
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]

        additional = super().sensors()

        sensors.extend(additional)
        return sensors

    def tick(self, input_data):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ["CAM_FRONT"]:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = Image.fromarray(img)

            # To convert image to numpy array, can use npimg = np.array(img)
            imgs[cam] = torch.from_numpy(np.array(self.rgb_prep(img)))

        gps = input_data["GPS"][1][:2]
        speed = input_data["SPEED"][1]["speed"]
        compass = input_data["IMU"][1][-1]
        acceleration = input_data["IMU"][1][:3]
        angular_velocity = input_data["IMU"][1][3:6]
        pos = self.gps_to_location(gps)

        pos_tf_agent = self.gps_to_carla_tf(input_data["GPS"][1])

        unfiltered_pos = pos

        if math.isnan(
            compass
        ):  # It can happen that the compass sends nan for a few frames
            # compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        compass = preprocess_compass(compass)

        if not self.filter_initialized:
            self.ukf.x = np.array(
                [pos_tf_agent[0], pos_tf_agent[1], normalize_angle(compass), speed]
            )
            self.filter_initialized = True

        try:
            self.ukf.predict(
                steer=self.last_control.steer,
                throttle=self.last_control.throttle,
                brake=self.last_control.brake,
            )
        except:
            self.ignore_filter = True

        if not self.ignore_filter:
            self.ukf.update(
                np.array(
                    [pos_tf_agent[0], pos_tf_agent[1], normalize_angle(compass), speed]
                )
            )
            # self.ukf2.update(np.array([pos_tf_agent2[0], pos_tf_agent2[1], normalize_angle(compass), speed]))
            filtered_state = self.ukf.x  # This is in carla coordinates
            filtered_gps = self.carla_tf_to_gps(
                [filtered_state[0], filtered_state[1], normalize_angle(compass)]
            )
            filtered_pos = self.gps_to_location(filtered_gps[:2])
        else:
            filtered_pos = unfiltered_pos

        (near_node, near_command), (far_node, far_command) = (
            self._route_planner.run_step(filtered_pos, True)
        )

        result = {
            "imgs": imgs,
            "gps": gps,
            "pos": filtered_pos,
            "speed": speed,
            "compass": compass,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "command_near": near_command,
            "command_near_xy": near_node,
            "command_far": far_command,
            "command_far_xy": far_node,
        }

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None, keep_ids=None):
        self.keep_ids = keep_ids

        self.step += 1
        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.last_control = control
            self.tick(input_data)
            return control

        tick_data = self.tick(input_data)

        self.context.append((tick_data))

        if len(self.context) > self.context_array_max_length:
            self.context.pop(0)

        if self.use_predicted_latent_with_gap:
            if len(self.context) < self.context_array_max_length:
                frc_context = {}
                frc_context["frc_speed"] = torch.zeros(1, 1, 1).to(
                    self.device, non_blocking=True
                )
                frc_context["frc_wps"] = torch.zeros(1, 1, 10, 2).to(
                    self.device, non_blocking=True
                )
                frc_context["frc_goal"] = (
                    torch.asarray([[5.1162534, -0.1575937], [26.005814, -0.09633584]])
                    .reshape(1, 1, 2, 2)
                    .to(self.device, non_blocking=True)
                )

                self.context[-1]["future_frc_context"] = {
                    "frc_rgb": self.context[0]["imgs"]["CAM_FRONT"]
                    .unsqueeze(0)
                    .unsqueeze(0),
                }

                frc_context["frc_rgb"] = self.get_cur_frc_context(self.context)

                tick_data["cur_frc_context"] = frc_context
            else:
                assert "future_frc_context" in self.context[0]

                frc_context = self.context[-1 - self.context_stride][
                    "future_frc_context"
                ]

                self.context[-1]["cur_frc_context"] = frc_context

        if tick_data["speed"] < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
            self.forced_move = False

        self.metric_info[self.step] = self.get_metric_info()

        if self.stuck_detector > self.cfg.creep_delay:
            self.forced_move = True
            self.stuck_detector = 0

        if (
            self.last_control.throttle == 0.0
            and tick_data["speed"] < 0.1
            and self.step % 5 != 0
            and len(self.control_preds) == 0
        ):
            if not self.forced_move and not self.use_predicted_latent_with_gap:
                print(
                    "Skipping control calculation because car is stopped and speed is very low. using last control as is."
                )
                return self.last_control
        else:
            if len(self.control_preds) == 0:
                if (
                    self.last_control.throttle == 0.0
                    and tick_data["speed"] < 0.1
                    and self.step % 5 != 0
                ):
                    print("Did not skip control because action delaying is active")

        control = self._get_control(tick_data)

        if len(self.control_preds) > 0:
            self.control_preds.append(control)
            control = self.control_preds.pop(0)

        print(len(self.control_preds), self.control_preds, control)

        self.last_control = control

        return control

    def _get_control(self, input_data):
        inp_dict = self.get_input_batch(input_data)

        inp_dict = {k: v.to(self.device) for k, v in inp_dict.items()}

        with torch.inference_mode():
            ot, labels = self.model(inp_dict, calculate_loss=False, return_labels=True)
        waypoints = ot["action"]["wps"].cumsum(dim=-2)[0].float()

        if self.use_predicted_latent_with_gap:
            future_frc_context = {}
            future_frc_context["frc_speed"] = inp_dict["state"][:1]
            future_frc_context["frc_wps"] = waypoints.unsqueeze(0).unsqueeze(0)
            future_frc_context["frc_goal"] = inp_dict["goal"][:1]
            future_frc_context["frc_rgb"] = (
                input_data["imgs"]["CAM_FRONT"].unsqueeze(0).unsqueeze(0)
            )

            input_data["future_frc_context"] = future_frc_context

        gt_velocity = torch.FloatTensor([input_data["speed"]]).unsqueeze(0)

        # waypoints
        is_stuck = False

        if self.forced_move:
            self.stuck_detector = 0
            self.forced_move_counter += 1
            if self.forced_move_counter < self.cfg.creep_duration:
                is_stuck = True
            else:
                self.forced_move_counter = 0
                self.forced_move = False

        steer, throttle, brake = self.control_pid(waypoints, gt_velocity, is_stuck)

        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # viz_trigger = (self.step % 25 == 0) and self.step < 2000
        viz_trigger = (
            self.step % self.cfg.viz_interval == 0
        ) and self.step < self.cfg.viz_max

        if viz_trigger and self.step > 2:
            visualize_trajectory_action_predictions(
                inp_dict,
                ot,
                labels=labels,
                save_dir=self.trajectory_viz_dir,
                model=self.model,
                save_prefix="plant",
                save_suffix=f"plant_{self.step}",
                save_idx=0,
                include_targets=False,  # Since we don't have targets at test time
                action_source="transformer-regression",
                visualize_gt=False,
            )

        return control

    def get_input_batch(self, input_data):
        inp_dict = {}

        speed = torch.FloatTensor([input_data["speed"]]).unsqueeze(0)
        inp_dict["state"] = speed.unsqueeze(0)

        inp_dict["rgb_front"] = (
            input_data["imgs"]["CAM_FRONT"].unsqueeze(0).unsqueeze(0)
        )

        if "target_point" in self.config.training["goal_type"]:
            ego_theta = input_data["compass"]
            command_near_xy = np.array(
                [
                    input_data["command_near_xy"][0] - input_data["pos"][0],
                    -input_data["command_near_xy"][1] + input_data["pos"][1],
                ]
            )
            rotation_matrix = np.array(
                [
                    [np.cos(ego_theta), -np.sin(ego_theta)],
                    [np.sin(ego_theta), np.cos(ego_theta)],
                ]
            )
            local_command_xy = rotation_matrix @ command_near_xy
            command_far_xy = np.array(
                [
                    input_data["command_far_xy"][0] - input_data["pos"][0],
                    -input_data["command_far_xy"][1] + input_data["pos"][1],
                ]
            )
            local_command_xy_far = rotation_matrix @ command_far_xy

            if "dual_target_point" in self.config.training["goal_type"]:
                local_command_xy_trch = torch.FloatTensor(
                    [local_command_xy[0], local_command_xy[1]]
                )

                if torch.norm(local_command_xy_trch) > 5.2:
                    local_command_xy_trch[0] = min(5.2, local_command_xy_trch[0])
                    local_command_xy_trch[1] = torch.sqrt(
                        max(
                            5.2**2 - local_command_xy_trch[0] ** 2,
                            torch.FloatTensor([0.0])[0],
                        )
                    ) * torch.sign(local_command_xy_trch[1])

                local_command_xy[0] = local_command_xy_trch[0]
                local_command_xy[1] = local_command_xy_trch[1]

                local_command_point = torch.from_numpy(
                    np.stack([local_command_xy, local_command_xy_far], axis=0)
                ).float()
            else:
                local_command_point = torch.FloatTensor(
                    [local_command_xy[0], local_command_xy[1]]
                )

                # If the distance is more than 6, shorten it to 6
                if torch.norm(local_command_point) > 6:
                    local_command_point = (
                        6 * local_command_point / torch.norm(local_command_point)
                    )

            local_command_point = local_command_point.reshape(1, -1, 2)
            goals = local_command_point

        inp_dict["goal"] = goals.unsqueeze(0)

        # Handle if context size is 1 where we don't have a previous action/reward
        action = torch.zeros((0, 8), dtype=torch.int)
        reward = torch.zeros((0, 1), dtype=torch.int)

        if self.use_predicted_latent_with_gap:
            assert "cur_frc_context" in input_data
            inp_dict["frc_wps"] = input_data["cur_frc_context"]["frc_wps"]
            inp_dict["frc_speed"] = input_data["cur_frc_context"]["frc_speed"]
            inp_dict["frc_goal"] = input_data["cur_frc_context"]["frc_goal"]
            if self.utilize_fast_current_latent:  # use current timestep rgb
                inp_dict["target_rgb_front"] = inp_dict["rgb_front"]
            inp_dict["rgb_front"] = input_data["cur_frc_context"]["frc_rgb"]

        inp_dict["reward"] = reward.unsqueeze(0)
        inp_dict["action"] = action.unsqueeze(0)

        return inp_dict

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (
            EARTH_RADIUS_EQUA * scale
        )
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = (
            scale
            * EARTH_RADIUS_EQUA
            * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0))
            - my
        )
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def gps_to_carla_tf(self, gps):
        gps = gps * np.asarray([111319.49082349832, 111319.49079327358, 1.0])

        return np.array([gps[1], -gps[0], gps[2]])

    def carla_tf_to_gps(self, carla_tf):
        carla_tf = np.array([-carla_tf[1], carla_tf[0], carla_tf[2]])
        gps = carla_tf / np.asarray([111319.49082349832, 111319.49079327358, 1.0])

        return gps

    def destroy(self):
        # Save metric dict
        import gzip

        with gzip.open(self.metrics_path + ".gz", "w") as f:
            f.write(json.dumps(self.metric_info).encode("utf-8"))

        super().destroy()
        if self.scenario_logger:
            self.scenario_logger.dump_to_json()
            del self.scenario_logger

        del self.model

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        waypoints = waypoints.data.cpu().numpy()

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        if is_stuck:
            desired_speed = (
                np.linalg.norm(waypoints[-2] - waypoints[-1]) * 2.0
            )  # Use speed from future timesteps

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

    def get_cur_frc_context(
        self, input_data, key1="future_frc_context", key2="frc_rgb"
    ):
        past_horizon = self.past_horizon_training

        length = len(input_data)

        indices = [
            max(length - 1 - ((i + 1) * self.context_stride), 0)
            for i in range(past_horizon + 1)
        ]

        frc_ctx = torch.cat(
            [input_data[idx][key1][key2] for idx in indices][::-1],
            dim=1,
        )

        return frc_ctx


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = []
        self.n = n
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        if len(self._window) > self.n:
            self._window.pop(0)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def preprocess_compass(compass):
    """
    Checks the compass for Nans and rotates it into the default CARLA coordinate
    system with range [-pi,pi].
    :param compass: compass value provided by the IMU, in radian
    :return: yaw of the car in radian in the CARLA coordinate system.
    """
    if math.isnan(compass):  # simulation bug
        compass = 0.0
    # The minus 90.0 degree is because the compass sensor uses a different
    # coordinate system then CARLA. Check the coordinate_sytems.txt file
    compass = normalize_angle(compass - np.deg2rad(90.0))

    return compass


# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model.
    # Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    """
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    """
    return vehicle_state


def state_mean(state, wm):
    """
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    """
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    """
    We use the arctan of the average of sin and cos of the angle to
    calculate the average of orientations.
    :param state: array of states to be averaged. First index is the
    timestep.
    """
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y
