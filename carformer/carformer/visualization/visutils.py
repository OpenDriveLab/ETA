from carformer.utils import TokenTypeIDs
from PIL import Image, ImageDraw, ImageFilter
import torch
import os
import numpy as np
import cv2


def extract_target_point_from_trajectory_goals(goals, goal_type):
    if "dual_target_point" in goal_type:
        goal_idx = goal_type.index("dual_target_point")
    elif "target_point" in goal_type:
        goal_idx = goal_type.index("target_point")
    else:
        raise ValueError(
            f"Unknown goal type {goal_type}. 'target_point' or 'dual_target_point' must be in the goal types."
        )

    return goals.reshape(-1, 2).numpy()


def draw_points_on_camera(
    camera,
    points,
    color=(0, 255, 255),
    first_origin=None,
    radius=4,
    blur=-1,
):
    rgb = Image.fromarray(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB))

    if blur > 0:
        # Start with a blank image and draw the points on it, blur it and then overlay it on the original image
        rgb_tmp = Image.new("RGBA", rgb.size, tuple([*color, 0]))
        draw = ImageDraw.Draw(rgb_tmp)
    else:
        draw = ImageDraw.Draw(rgb)

    for i, pt in enumerate(points):
        x, y = pt[:2].astype(int)

        y = first_origin[-1] + y
        x = first_origin[0] + x
        if x < 0 or x >= rgb.size[0] or y < 0 or y >= rgb.size[-1]:
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

        # point_list = first_origin + point_list * pix_per_meter

        # point_list[0] += i * skip_size

        # for j, point in enumerate(point_list):
        #     cv2.circle(
        #         img=canvas,
        #         center=tuple(point.astype(np.int32)),
        #         radius=np.rint(radius).astype(int),
        #         color=tuple(c * (color_decay**j) for c in color),
        #         thickness=cv2.FILLED,
        #     )

    if blur > 0:
        rgb_tmp = rgb_tmp.filter(ImageFilter.GaussianBlur(blur))
        if rgb.mode == "RGB":
            rgb = rgb.convert("RGBA")

        rgb = Image.alpha_composite(rgb, rgb_tmp)

        # convert back to RGB
        rgb = rgb.convert("RGB")

    # draw.ellipse([first_origin[0]-radius, first_origin[-1]-radius, first_origin[0]+radius, first_origin[-1]+radius], fill=color)

    # Return cv2 format array
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def get_action_canvas(
    action,
    action_types,
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
    supp_action=None,
):
    if "path" in action_types:
        return get_action_waypoints_path(
            action,
            action_types,
            size=size,
            bev_canvas=bev_canvas,
            copy_bev_canvas=copy_bev_canvas,
            bev_crop_type=bev_crop_type,
            pix_per_meter=pix_per_meter,
            supp_action=supp_action,
        )
    else:
        raise ValueError(
            f"Unknown action type {action_types}. 'path' must be in the action types."
        )


def get_action_waypoints_path(
    action,
    action_types,
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
    supp_action=None,
):
    if bev_canvas is None:
        bev_canvas = np.zeros((size, size * action.shape[0], 3), dtype=np.uint8)
    else:
        if copy_bev_canvas:
            bev_canvas = bev_canvas.copy()

    waypoints = action[0, 20:].copy()

    waypoints[:, -1] *= -1

    path = action[0, :20]

    waypoints = point_to_canvas_coordinates_rel_to_center(waypoints, height=-1.6) / 2
    path = point_to_canvas_coordinates_rel_to_center(path, height=-1.6) / 2

    if bev_crop_type == "dualfront":
        origin = (size, size // 2)
    elif bev_crop_type == "front":
        origin = (size // 2, size)
    else:
        origin = (size // 2, size // 2)

    bev_canvas = draw_points_on_camera(
        bev_canvas, waypoints, color=(0, 0, 255), first_origin=origin, radius=5
    )

    bev_canvas = draw_points_on_camera(
        bev_canvas, path, color=(255, 255, 0), first_origin=origin, radius=2
    )
    if supp_action is not None:
        supp_waypoints = supp_action[0, 20:].copy()
        supp_waypoints[:, -1] *= -1
        supp_path = supp_action[0, :20]

        supp_waypoints = (
            point_to_canvas_coordinates_rel_to_center(supp_waypoints, height=-1.6) / 2
        )
        supp_path = (
            point_to_canvas_coordinates_rel_to_center(supp_path, height=-1.6) / 2
        )

        bev_canvas = draw_points_on_camera(
            bev_canvas, supp_waypoints, color=(0, 0, 155), first_origin=origin, radius=3
        )
        bev_canvas = draw_points_on_camera(
            bev_canvas, supp_path, color=(155, 0, 155), first_origin=origin, radius=1
        )

    if not copy_bev_canvas:
        raise ValueError("Not implemented")
    return bev_canvas


def point_to_canvas_coordinates_rel_to_center(
    points, original_size=(1600, 800), height=-1
):  # shape: Nx2
    rgb_front_intrinsic = np.asarray(
        [
            [1142.5184053936916, 0.0, 800.0],
            [0.0, 1142.5184053936916, 450.0],
            [0.0, 0.0, 1.0],
        ]
    )

    points = np.stack(
        [points[:, 1], np.ones_like(points[:, 0]) * height, points[:, 0]], axis=-1
    )
    # points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # If any of points[:, 2:3] are between 1e-4 and -1e-4, then we set them to plus 1e-4
    mask = np.logical_and(points[:, 2:3] < 1e-4, points[:, 2:3] > -1e-4)

    points[:, 2:3][mask] = 1e-4

    points = points / points[:, 2:3]
    points = (rgb_front_intrinsic @ points.T[:3, :]).T

    points = -points

    points[:, 0] += original_size[0] // 2
    points[:, 1] += original_size[1] // 2

    return points


def get_goal_canvas(
    commands,
    command_types,
    size=192,
    color=(255, 0, 255),
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if "command" in command_types:
        return get_command_canvas(commands, size=size, color=color)
    else:
        return get_targetpoint_canvas(
            commands,
            command_types,
            size=size,
            color=color,
            bev_canvas=bev_canvas,
            copy_bev_canvas=copy_bev_canvas,
            bev_crop_type=bev_crop_type,
            pix_per_meter=pix_per_meter,
        )


def get_targetpoint_canvas(
    commands,
    command_types,
    color=(255, 0, 255),
    size=192,
    bev_canvas=None,
    copy_bev_canvas=False,
    bev_crop_type="front",
    pix_per_meter=5,
):
    if bev_canvas is None:
        bev_canvas = np.zeros((size, size * commands.shape[0], 3), dtype=np.uint8)
    else:
        if copy_bev_canvas:
            bev_canvas = bev_canvas.copy()

    # For now, assert size of action is 8 and no other action types
    assert len(command_types) == 1 and (
        command_types[0] == "target_point" or command_types[0] == "dual_target_point"
    )

    target_points = extract_target_point_from_trajectory_goals(commands, command_types)
    # print(target_points)
    target_points = (
        point_to_canvas_coordinates_rel_to_center(target_points, height=-1.6) / 2
    )

    # print(target_points)
    # canvas = draw_points_on_camera(canvas, target_points, color=(255, 0, 255), first_origin=origin, radius=10, blur=8)

    if bev_crop_type == "dualfront":
        origin = (size, size // 2)
    elif bev_crop_type == "front":
        origin = (size // 2, size)
    else:
        origin = (size // 2, size // 2)

    # Draw the target points
    bev_canvas = draw_points_on_camera(
        bev_canvas, target_points, color=color, first_origin=origin, radius=12, blur=8
    )

    if not copy_bev_canvas:
        return None
    return bev_canvas


def visualize_trajectory_action_predictions(
    batch,
    batch_outputs,
    save_dir=None,
    labels=None,
    save_idx=0,
    action_source="transformer-regression",
    save_suffix="",
    do_write=True,
    visualize_gt=True,
    *args,
    **kwargs,
):
    # Move batch to cpu
    # batch = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    # Copy batch shallowly
    batch = {k: v for k, v in batch.items()}

    # Convert labels to empty dict if None
    if labels is None:
        labels = {}

    if action_source == "transformer":
        if "output_dict" in batch_outputs:
            batch_outputs = batch_outputs["output_dict"]
        batch_outputs = {k: v for k, v in batch_outputs.items()}

        # Change the batch action to be the predicted action
        pred_actions = batch_outputs[TokenTypeIDs.ACTION].to(batch["action"].device)
    elif action_source == "gru":
        if "waypoints" in batch_outputs:
            pred_actions = batch_outputs["waypoints"].to(batch["action"].device)
        else:
            pred_actions = batch_outputs.to(batch["action"].device)
    elif action_source == "transformer-regression":
        waypoints = batch_outputs["action"]["wps"].float().detach()  # 1x10x2
        path = batch_outputs["action"]["path"].float().detach()  # 1x20x2

        waypoints = waypoints.cumsum(-2)
        path = path.cumsum(-2)
        if len(waypoints.shape) == 3:
            waypoints = waypoints.unsqueeze(1)
            path = path.unsqueeze(1)

        if visualize_gt:
            supp_action = batch["action"]  # [save_idx].unsqueeze(0)
        else:
            supp_action = None

        batch["action"] = torch.zeros(
            (waypoints.shape[0], 1, 30, 2), dtype=path.dtype, device=path.device
        )
        batch["action"][:, :, :20, :] = path
        batch["action"][:, :, 20:, :] = waypoints
    else:
        raise ValueError(f"Unknown action source {action_source}")

    # pred_actions_len = pred_actions.shape[0]

    # batch["action"] = torch.cat(
    #     (
    #         batch["action"],
    #         pred_actions.reshape(
    #             batch["action"].shape[0], -1, batch["action"].shape[-1]
    #         ),
    #     ),
    #     dim=1,
    # )
    # import ipdb; ipdb.set_trace()
    # return visualize_trajectory(batch, save_dir, save_idx=save_idx, *args, **kwargs)
    return visualize_input_from_batch(
        batch=batch,
        batch_idx=save_idx,
        batch_outputs=batch_outputs,
        labels=labels,
        save_dir=save_dir,
        save_affix=f"pred_" + "{}".format(save_suffix),
        model=kwargs.get("model", None),
        save_prefix=None,
        do_write=do_write,
        supp_action=supp_action,
    )


def get_bev_canvas(
    batch,
    batch_idx,
    model=None,
    batch_outputs=None,
    labels=None,
    include_targets=True,
    use_target_mask_if_available=True,
):
    if model is not None:
        if "rgb_front" in batch:
            bev_mode = "rgb_front"
        else:
            raise ValueError(
                "Model is provided but no rgb_front in batch. "
                "This is unexpected, please check your model and batch."
            )
    else:
        if "rgb_front" in batch:
            bev_mode = "rgb_front"
        else:
            raise ValueError("No rgb in batch")

    targets_rgb = None
    if bev_mode == "rgb_front":
        # shape: BxTxHxWx3
        if use_target_mask_if_available and "goal_mask" in batch:
            rgb_front = batch["goal_mask"]
        else:
            rgb_front = batch["rgb_front"]

        gt_rgb = (
            rgb_front[batch_idx]
            .cpu()
            .numpy()
            .transpose((0, 1, 2, 3))
            .astype(np.float32)
            / 255.0
        )

        gt_rgb = gt_rgb[:1].reshape(-1, *gt_rgb.shape[2:])
        gt_reproduced = None
        preds_rgb = None
        if include_targets:
            if "target_rgb_front" in batch:
                targets_rgb = (
                    batch["target_rgb_front"][batch_idx]
                    .cpu()
                    .numpy()
                    .transpose((1, 0, 2, 3))
                    .astype(np.float32)
                    / 255.0
                )
                targets_rgb = targets_rgb.squeeze(1)

                waypoints = batch["frc_wps"][batch_idx, 0].cpu().numpy().copy()

                waypoints[:, -1] *= -1

                waypoints_wrld = (
                    point_to_canvas_coordinates_rel_to_center(waypoints, height=-1.6)
                    / 2
                )

                speed = batch["frc_speed"][batch_idx, 0].cpu().item()
                goals = batch["frc_goal"][batch_idx, 0].cpu()
                goals = (
                    point_to_canvas_coordinates_rel_to_center(goals, height=-1.6) / 2
                )

                targets_rgb_pil = Image.fromarray((targets_rgb * 255).astype(np.uint8))
                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(targets_rgb_pil)

                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/dejavu/DejaVuSans.ttf", 15
                    )
                except IOError:
                    font = ImageFont.load_default()

                # Get center of the image
                width, height = targets_rgb_pil.size

                # Calculate the position for the text
                position = (0, height // 2)
                # if x > y:
                #     break
                # Draw the text at the calculated position with the specified angle
                draw.text(
                    position,
                    f"Speed: {speed:.2f}\n Waypoints: {waypoints}\n Goals: {goals}",
                    font=font,
                    fill=(255, 255, 255, 128),
                    anchor="lm",
                )

                targets_rgb = np.array(targets_rgb_pil)

                draw_points_on_camera(
                    targets_rgb,
                    waypoints_wrld,
                    color=(255, 0, 0),
                    first_origin=(width // 2, height // 2),
                    radius=4,
                )
                draw_points_on_camera(
                    targets_rgb,
                    goals,
                    color=(200, 200, 0),
                    first_origin=(width // 2, height // 2),
                    radius=2,
                )

                targets_rgb = targets_rgb.astype(np.float32) / 255.0

    if gt_reproduced is not None:
        gt_reproduced = gt_reproduced.reshape(
            gt_reproduced.shape[0], -1, gt_reproduced.shape[-1]
        )

    rows = [x for x in [gt_rgb, gt_reproduced, targets_rgb, preds_rgb] if x is not None]
    canvas_unit_size = gt_rgb.shape[0]

    canvas = np.zeros((canvas_unit_size * len(rows), gt_rgb.shape[1], 3))

    canvas = np.concatenate(rows, axis=0)

    # Convert canvas of 0-1 floats to 0-255 uint8
    canvas = (canvas * 255).astype(np.uint8)
    # Rgb to bgr
    canvas = canvas[:, :, ::-1]

    return canvas, canvas_unit_size


def visualize_input_from_batch(
    batch,
    batch_idx,
    batch_outputs,
    labels,
    save_dir,
    save_affix,
    model,
    save_prefix,
    do_write=True,
    supp_action=None,
    use_target_mask_if_available=True,
):
    canvas, img_size = get_bev_canvas(
        batch=batch,
        batch_idx=batch_idx,
        model=model,
        batch_outputs=batch_outputs,
        labels=labels,
        use_target_mask_if_available=use_target_mask_if_available,
    )

    canvas_to_reuse = canvas

    # Add a black copy of the canvas
    black = np.zeros_like(canvas_to_reuse)

    canvas_to_reuse = np.concatenate([canvas_to_reuse, black], axis=0)

    actions = batch["action"][batch_idx].cpu().numpy()
    if supp_action is not None:
        supp_action = supp_action[batch_idx].cpu().numpy()

    action_canvas = get_action_canvas(
        actions,
        ["path", "waypoints"],
        size=img_size,
        bev_canvas=canvas_to_reuse,
        copy_bev_canvas=True,
        bev_crop_type="dualfront",
        pix_per_meter=3,
        supp_action=supp_action,
    )

    goal = batch["goal"][batch_idx].cpu()
    goal_canvas = get_goal_canvas(
        goal,
        ["target_point"],
        size=img_size,
        bev_canvas=action_canvas,
        copy_bev_canvas=True,
        bev_crop_type="dualfront",
        pix_per_meter=3,
    )

    if not do_write:
        return goal_canvas

    impath = os.path.join(
        save_dir,
        f"{save_prefix}_predictions" if save_prefix else "",
        "epoch_{}.png".format(save_affix),
    )

    # Save as epoch_{epoch}.png in the log directory
    cv2.imwrite(
        impath,
        goal_canvas,
    )

    return impath
