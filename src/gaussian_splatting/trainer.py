from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from dreifus.matrix import Pose, Intrinsics
from dreifus.util.visualizer import ImageWindow
from elias.config import Config
from faiss import IndexFlatL2
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render, render_gsplat, render_radegs
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.scene.gaussian_model import GSplatModel
from gaussian_splatting.scene.gaussian_model_radegs import GaussianModelRaDeGS
from gaussian_splatting.utils.graphics_utils import depth_double_to_normal, point_double_to_normal
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, l1_loss_appearance
from gaussian_splatting.utils.viewer import GaussianViewer
from topo.util.logging import LoggerBundle
from tqdm import tqdm


@dataclass
class GSOptimizationConfig(Config):
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False

    use_densification: bool = True
    use_opacity_reset: bool = True

    # Regularizations
    reg_from_iter: int = 15_000  # Only apply regularizations after that many iterations
    lambda_pos_reg: float = 0  # Keep Gaussians close to initialization
    lambda_scale_reg: float = 0  # Keep Gaussian scales small
    lambda_depth_normal: float = 0  # RaDe-GS: Apply depth distortion and normals consistency
    use_coord_map: bool = False  # RaDe-GS: Apply distortion on coordinate map
    disable_filter3D: bool = True  # RaDe-GS: Do not compute scales from cameras
    appearance_embeddings_lr: float = 0.001
    appearance_network_lr: float = 0.001
    use_decoupled_appearance: bool = False  # RaDe-GS: Use screen-space CNN to account for view-inconsistencies in GT


def run_gaussian_splatting(gaussian_model: Union[GaussianModel, GSplatModel, GaussianModelRaDeGS],
                           cam_2_world_poses: List[Pose],
                           intrinsics: List[Intrinsics],
                           images: List[np.ndarray],
                           bg_color: torch.Tensor = torch.Tensor([1., 1., 1.]),
                           config: GSOptimizationConfig = GSOptimizationConfig(),
                           device: torch.device = torch.device('cuda'),
                           use_viewer: bool = False,
                           use_visualization_window: bool = False,
                           visualize_idx: int = 0,
                           logger_bundle: LoggerBundle = LoggerBundle()):
    gaussian_model.training_setup(config)
    optimizer = gaussian_model.optimizer
    use_radegs = isinstance(gaussian_model, GaussianModelRaDeGS)
    use_gsplat = isinstance(gaussian_model, GSplatModel)

    # ==========================================================
    # Viewer
    # ==========================================================

    # Setup interactive web-based 3D Viewer for 3D Gaussians
    if use_viewer:
        viewer = GaussianViewer(gaussian_model,
                                poses=cam_2_world_poses,
                                intrinsics=intrinsics,
                                images=images,
                                size=0.03)
        viewer.server.set_up_direction("-y")

    image_buffer = None
    if use_visualization_window:
        image_buffer = np.zeros_like(images[0], dtype=np.float32)
        image_window = ImageWindow(image_buffer)

    # ==========================================================
    # Initialization
    # ==========================================================

    # Initialize RaDe-GS 3D filters
    train_cameras = None
    if use_radegs:
        if config.disable_filter3D:
            gaussian_model.reset_3D_filter()  # TODO: Does it make a difference to use compute_3D_filter()?
        else:
            H, W, _ = images[0].shape
            train_cameras = [pose_to_rendercam(pose, intr, W, H) for pose, intr in zip(cam_2_world_poses, intrinsics)]
            gaussian_model.compute_3D_filter(train_cameras)

    # Initialize NN lookup structures for position regularization
    vertex_index = None
    initial_point_positions = None
    if config.lambda_pos_reg > 0:
        initial_point_positions = gaussian_model.get_xyz.clone()
        vertex_index = IndexFlatL2(3)
        vertex_index.add(np.ascontiguousarray(initial_point_positions.detach().cpu().numpy()))

    # ==========================================================
    # Optimization Loop
    # ==========================================================

    bg_color = torch.tensor(bg_color, device=device, dtype=torch.float32)
    torch_images = [torch.tensor(image / 255, device=device, dtype=torch.float32).permute(2, 0, 1)
                    for image in images]
    progress = tqdm(range(config.iterations), miniters=100, position=-1, maxinterval=float('inf'))
    for iteration in progress:
        sample_idx = np.random.randint(len(torch_images))
        gt_image = torch_images[sample_idx]
        cam_2_world_pose = cam_2_world_poses[sample_idx]
        single_intrinsics = intrinsics[sample_idx]
        gs_camera = pose_to_rendercam(cam_2_world_pose, single_intrinsics, gt_image.shape[2], gt_image.shape[1])

        # Render current 3D reconstruction
        use_regularization = iteration >= config.reg_from_iter
        if use_gsplat:
            output = render_gsplat(gs_camera, gaussian_model, bg_color)
        elif use_radegs:
            output = render_radegs(gs_camera, gaussian_model, PipelineParams2(), bg_color,
                                   require_coord=use_regularization and config.use_coord_map,
                                   require_depth=use_regularization and not config.use_coord_map)
        else:
            output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color)
        rendered_image = output['render']

        # Compute L1 loss
        if use_radegs and config.use_decoupled_appearance:
            l1 = l1_loss_appearance(rendered_image, gt_image, gaussian_model, sample_idx)
        else:
            l1 = l1_loss(rendered_image, gt_image)

        # SSIM loss
        ssim_loss = ssim(rendered_image, gt_image)
        loss = (1.0 - config.lambda_dssim) * l1 + config.lambda_dssim * (1.0 - ssim_loss)

        # Position regularization
        if config.lambda_pos_reg > 0:
            gaussian_positions = gaussian_model.get_xyz
            nearest_vertex_idxs = vertex_index.search(
                np.ascontiguousarray(gaussian_positions.detach().cpu().numpy()), 1)[1][:, 0]
            closest_vertex_distance = gaussian_positions - initial_point_positions[nearest_vertex_idxs]
            closest_vertex_distance = closest_vertex_distance.norm(dim=-1).square().mean()
            loss = loss + config.lambda_pos_reg * closest_vertex_distance

            logger_bundle.log_metrics({
                "train/losses/closest_vertex_distance": closest_vertex_distance.item(),
            }, step=iteration)

        # Scale regularization
        if config.lambda_scale_reg > 0:
            gaussian_scales = gaussian_model.get_scaling
            scale_threshold = 4
            scale_mask = gaussian_scales > scale_threshold
            if scale_mask.any():
                scale_reg = (gaussian_scales[scale_mask] - scale_threshold).square().mean()
                loss = loss + config.lambda_scale_reg * scale_reg

                logger_bundle.log_metrics({
                    "train/losses/scale_reg": scale_reg.item(),
                }, step=iteration)

        # RaDe-GS: Depth Distortion and Normal Consistency
        if config.lambda_depth_normal > 0 and use_regularization:
            if not config.use_coord_map:
                rendered_expected_depth: torch.Tensor = output["expected_depth"]
                rendered_median_depth: torch.Tensor = output["median_depth"]
                rendered_normal: torch.Tensor = output["normal"]
                depth_middepth_normal = depth_double_to_normal(gs_camera, rendered_expected_depth,
                                                               rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = output["expected_coord"]
                rendered_median_coord: torch.Tensor = output["median_coord"]
                rendered_normal: torch.Tensor = output["normal"]
                depth_middepth_normal = point_double_to_normal(gs_camera, rendered_expected_coord,
                                                               rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1 - depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[
                1].mean()

            loss = loss + config.lambda_depth_normal * depth_normal_loss
            logger_bundle.log_metrics({
                "train/losses/depth_normal": depth_normal_loss.item(),
            }, step=iteration)

        # TODO: ADD beta loss
        # TODO: ADD TV loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Gaussian Splatting Training tricks:
        #  1. Learning rate schedule for 3D positions (positions will receive smaller and small updates)
        gaussian_model.update_learning_rate(iteration)

        if (config.use_densification or config.use_opacity_reset) and iteration < config.densify_until_iter:
            # Important, otherwise prune and densify don't work
            visibility_filter = output['visibility_filter']
            gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter],
                                                                      output['radii'][visibility_filter])
            if use_gsplat:
                gaussian_model.add_densification_stats(output['viewspace_points'], visibility_filter, gt_image.shape[2],
                                                       gt_image.shape[1])
            else:
                gaussian_model.add_densification_stats(output['viewspace_points'], visibility_filter)

            #  2. Adaptive density control:
            #       - Clone/Split Gaussians with high positional gradient (that move around a lot)
            #       - Prune Gaussians with low opacity (don't clutter scene)
            if config.use_densification and iteration > config.densify_from_iter and (
                    iteration + 1) % 100 == 0:  # or (dataset.white_background and iteration == opt.densify_from_iter)
                gaussian_model.densify_and_prune(config.densify_grad_threshold, 0.005, 1, None)

                if use_radegs:
                    if config.disable_filter3D:
                        gaussian_model.reset_3D_filter()
                    else:
                        gaussian_model.compute_3D_filter(cameras=train_cameras)

            #  3. Opacity reset (needed such that Pruning is more effective)
            if config.use_opacity_reset and (iteration + 1) % 3000 == 0:
                gaussian_model.reset_opacity()

            if use_radegs and (
                    iteration + 1) % 100 == 0 and iteration > config.densify_until_iter and not config.disable_filter3D:
                if iteration < config.iterations - 100:
                    # don't update in the end of training
                    gaussian_model.compute_3D_filter(cameras=train_cameras)

        # Log prediction to visualization window
        if use_visualization_window and sample_idx == visualize_idx:
            image_buffer[:] = rendered_image.permute(1, 2, 0).detach().cpu().numpy()

        logger_bundle.log_metrics({
            "train/losses/l1": l1.item(),
            "train/losses/ssim": ssim_loss.item(),
            "train/metrics/n_gaussians": gaussian_model._xyz.shape[0]
        }, step=iteration)

        progress.set_postfix({"loss": loss.item()}, refresh=False)
