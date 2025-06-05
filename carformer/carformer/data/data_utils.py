import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop


def get_rgb_preprocessing_function_from_config(config):
    return get_rgb_preprocessing_function(
        config.rgb_crop.type,
        config.rgb_crop.crop_size,
        config.rgb_crop.resize,
    )


def get_rgb_preprocessing_function(crop_type, crop_size, resize):
    transforms = []

    if crop_type == "dualcenter":
        # Width: 2xcrop_size, height: crop_size
        # Then resize to 2xresize, resize
        transforms.append(CenterCrop((crop_size, 2 * crop_size)))
        if resize > 0:
            transforms.append(Resize((resize, 2 * resize)))
    elif crop_type == "center":
        transforms.append(CenterCrop(crop_size))
        if resize > 0:
            transforms.append(Resize(resize))
    else:
        raise NotImplementedError

    transform = Compose(transforms)

    return transform


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""
    vehicle_matrix = np.array(waypoints[0])
    for i in range(
        0, len(waypoints)
    ):  # TODO: Start from 1 because 0 is ego vehicle initial position
        matrix = np.array(waypoints[i])
        waypoints[i] = vehicle_matrix @ np.linalg.inv(matrix)

    return waypoints


########################################################################

############## WARPING

###################################################################3####


def compute_relative_transform(origin, current, pix_per_meter=5):
    result = torch.bmm(torch.linalg.inv(origin), (current))

    return result


def get_affine_grid_transform(origin, current, inp_size=400, pix_per_meter=5):
    relative_transform = compute_relative_transform(origin, current, pix_per_meter)

    translation = relative_transform[:, :2, 3:] / ((inp_size / 2) / pix_per_meter)
    translation[:, [0, 1]] = translation[:, [1, 0]]

    affine_grid_transform = torch.cat(
        (torch.transpose(relative_transform[:, :2, :2], 2, 1), translation), axis=2
    )

    # rot x, y. dont take height.
    # affine_grid_transform = torch.from_numpy(affine_grid_transform).float()

    return affine_grid_transform


def warp_sequence(x, ego_matrices, mode="nearest", spatial_extent=None):
    """
    Batch-compatible warping function.

    Warps a sequence based on the first frame using ego vehicle transformation matrices.
    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    out = [x[:, 0]]

    # print('X. SHAPE ', x.shape)
    base_frame = ego_matrices[:, 0]  # torch.from_numpy()

    for t in range(1, sequence_length):
        curr_frame = ego_matrices[:, t]  # torch.from_numpy()
        aff_grid = get_affine_grid_transform(
            base_frame, curr_frame, inp_size=x.shape[-1], pix_per_meter=5
        )  # .unsqueeze(0)

        grid = torch.nn.functional.affine_grid(
            aff_grid, size=x[:, 0].shape, align_corners=False
        )

        warped_bev = torch.nn.functional.grid_sample(
            (x[:, t]),
            grid.float(),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        out.append(warped_bev)

    return torch.stack(out, 1)


### Forecast utils
def extract_forecast_targets(
    timesteps,
    context_length,
    future_horizon,
    forecast_steps=1,
    use_slots=True,
    object_level=True,
):
    assert len(timesteps) == context_length + future_horizon
    for i in range(context_length):
        timesteps[i].state["target_rgb_front"] = timesteps[i + forecast_steps].state[
            "rgb_front"
        ]


def circle_line_segment_intersection(
    circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9
):
    """Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.
                      False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a
                        tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the
                                           circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    Credit: https://stackoverflow.com/a/59582674/9173068
    """

    if np.linalg.norm(pt1 - pt2) < 0.000000001:
        # print('Problem')
        return []

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        # This makes sure the order along the segment is correct
        # intersections = [(cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr**2,
        #                   cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr**2)
        #                  for sign in ((1, -1) if dy < 0 else (-1, 1))]

        # Write explicitly to avoid iteration
        if dy < 0:
            sign_1 = 1
            sign_2 = -1
        else:
            sign_1 = -1
            sign_2 = 1

        intersections = [
            (
                cx + (big_d * dy + sign_1 * sign_2 * dx * discriminant**0.5) / dr**2,
                cy + (-big_d * dx + sign_1 * abs(dy) * discriminant**0.5) / dr**2,
            ),
            (
                cx + (big_d * dy + sign_2 * sign_2 * dx * discriminant**0.5) / dr**2,
                cy + (-big_d * dx + sign_2 * abs(dy) * discriminant**0.5) / dr**2,
            ),
        ]

        if (
            not full_line
        ):  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy
                for xi, yi in intersections
            ]
            intersections = [
                pt
                for pt, frac in zip(intersections, fraction_along_segment)
                if 0 <= frac <= 1
            ]
        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections


import numpy as np


def iterative_line_interpolation(route, num_points=20):
    if not isinstance(route, np.ndarray):
        route = np.array(route)

    interpolated_route_points = []

    min_distance = 0.5
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = route[current_route_index]

    while len(interpolated_route_points) < num_points:
        # First point should be min_distance away from the vehicle.
        dist = np.linalg.norm(current_point - last_interpolated_point)
        if dist < min_distance:
            current_route_index += 1
            last_point = current_point

        if current_route_index < route.shape[0]:
            current_point = route[current_route_index]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
                full_line=False,
            )

        else:  # We hit the end of the input route. We extrapolate the last 2 points
            current_point = route[-1]
            last_point = route[-2]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
                full_line=True,
            )

        # 3 cases: 0 intersection, 1 intersection, 2 intersection
        if len(intersection) > 1:  # 2 intersections
            # Take the one that is closer to current point
            point_1 = np.array(intersection[0])
            point_2 = np.array(intersection[1])
            direction = current_point - last_point
            dot_p1_to_last = np.dot(point_1, direction)
            dot_p2_to_last = np.dot(point_2, direction)

            if dot_p1_to_last > dot_p2_to_last:
                intersection_point = point_1
            else:
                intersection_point = point_2
            add_point = True
        elif len(intersection) == 1:  # 1 Intersections
            intersection_point = np.array(intersection[0])
            add_point = True
        else:  # 0 Intersection
            add_point = False

        if add_point:
            last_interpolated_point = intersection_point
            interpolated_route_points.append(intersection_point)
            min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

    interpolated_route_points = np.array(interpolated_route_points)

    return interpolated_route_points


def interpolate_waypoints(wps, num_points=10):
    assert len(wps) > 1, "Need at least 2 waypoints to interpolate"
    last_vector = wps[-1] - wps[-2]

    if len(wps) >= num_points:
        return wps[:num_points]

    interpolated_points = []
    for i in range(len(wps) - num_points):
        # interpolated_points.append(wps[i] + last_vector * (i + 1))
        interpolated_points.append((-99999, -99999))

    return np.concatenate([wps, np.stack(interpolated_points)], axis=0)


class iterative_intepolator:
    def __init__(self, num_points=20):
        self.num_points = num_points
        self.min_distance = 0.5
        self.last_interpolated_point = np.array([0.0, 0.0])
        self.current_route_index = 0
        self.current_point = None
        self.last_point = None
        self.interpolated_route_points = []
        self.last_inputs = []

    def __call__(self, point):
        if self.current_point is None:
            self.current_point = point
            self.last_point = point

        if point is not None:
            self.last_inputs.append(point)
            self.last_inputs = self.last_inputs[-2:]

            dist = np.linalg.norm(self.current_point - self.last_interpolated_point)
            if dist < self.min_distance:
                # self.current_route_index += 1
                self.last_point = self.current_point
                return len(self)

            self.current_point = np.asarray(point)
            intersection = circle_line_segment_intersection(
                circle_center=self.last_interpolated_point,
                circle_radius=self.min_distance,
                pt1=self.last_point,
                pt2=self.current_point,
                full_line=False,
            )
        else:
            current_point = self.last_inputs[-1]
            last_point = self.last_inputs[-2]
            intersection = circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
                full_line=True,
            )

        if len(intersection) > 1:  # 2 intersections
            # Take the one that is closer to current point
            point_1 = np.array(intersection[0])
            point_2 = np.array(intersection[1])
            direction = current_point - last_point
            dot_p1_to_last = np.dot(point_1, direction)
            dot_p2_to_last = np.dot(point_2, direction)

            if dot_p1_to_last > dot_p2_to_last:
                intersection_point = point_1
            else:
                intersection_point = point_2
        elif len(intersection) == 1:  # 1 Intersections
            intersection_point = np.array(intersection[0])
        else:  # 0 Intersection
            return len(self)

        self.last_interpolated_point = intersection_point
        self.interpolated_route_points.append(intersection_point)
        self.min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

    def get_interpolated_points(self):
        if len(self.interpolated_route_points) < self.num_points:
            for i in range(len(self.interpolated_route_points), self.num_points):
                self(None)

                print(len(self))

        return self.interpolated_route_points

    def __len__(self):
        return self.num_points


# Adapted from https://github.com/OpenDriveLab/TCP/blob/9ec4db0f0424801cdd607f1de930290830c5e88e/leaderboard/team_code/auto_pilot.py#L339
def get_hazard_directions(vehicle_list):
    ego_vehicles = [x for x in vehicle_list if x["class"] == "ego_vehicle"]

    if len(ego_vehicles) == 0:
        return []

    if len(ego_vehicles) > 1:
        print("More than one ego vehicle found")
        return []

    ego_vehicle = ego_vehicles[0]

    z = ego_vehicle["location"][1]

    o1 = _orientation(ego_vehicle["rotation"][-1])
    p1 = np.asarray(ego_vehicle["location"][:2])
    s1 = max(2, 3.0 * ego_vehicle["speed"])  # increases the threshold distance
    v1_hat = o1
    v1 = s1 * v1_hat

    hazard_directions = []

    for target_vehicle in vehicle_list:
        if target_vehicle["class"] == "ego_vehicle":
            continue

        if target_vehicle.get("base_type", None) != "car":
            continue

        o2 = _orientation(target_vehicle["rotation"][-1])
        p2 = np.asarray(target_vehicle["location"][:2])
        s2 = max(5.0, 2.0 * target_vehicle["speed"])
        v2_hat = o2
        v2 = s2 * v2_hat

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))

        angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

        # print()
        angle_from_ego = np.degrees(np.arccos(v2_hat.dot(p2_p1_hat)))

        # to consider -ve angles too
        angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
        angle_between_heading = min(
            angle_between_heading, 360.0 - angle_between_heading
        )

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        # print(target_vehicle["type_id"], target_vehicle["distance"], target_vehicle["color"])
        # print("s1", s1, "dist", distance, "tgt_dist", target_vehicle["distance"])
        # print(angle_to_car, angle_between_heading, distance, s1)
        # print("angle from ego", angle_from_ego)

        hazard_directions.append(angle_from_ego)

    return hazard_directions


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 4)  # how many seconds until collision

    return collides, p1 + x[0] * v1


def is_walker_hazard(objects_list):
    ego_vehicles = [x for x in objects_list if x["class"] == "ego_vehicle"]
    if len(ego_vehicles) == 0:
        print("No ego vehicle found")
        return False

    ego_vehicle = ego_vehicles[0]

    z = ego_vehicle["location"][-1]

    walkers = [x for x in objects_list if x["class"] == "walker"]
    p1 = np.asarray(ego_vehicle["location"][:2])
    v1 = 10.0 * _orientation(ego_vehicle["rotation"][-1])

    for walker in walkers:
        v2_hat = _orientation(walker["rotation"][-1])
        s2 = walker["speed"]

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + np.asarray(walker["location"][:2])
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            return True

    return False
