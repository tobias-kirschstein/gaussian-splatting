#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from typing import Dict

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_features import GaussianRasterizer as GaussianFeatureRasterizer
from diff_gaussian_rasterization_radegs import GaussianRasterizer as GaussianRasterizerRaDeGS
from diff_gaussian_rasterization_radegs import GaussianRasterizationSettings as GaussianRasterizationSettingsRaDeGS
from diff_gaussian_rasterization_distwar import GaussianRasterizer as DistwarGaussianRasterizer
from diff_gaussian_rasterization_distwar_features import GaussianRasterizer as DistwarGaussianFeatureRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.gaussian_model_radegs import GaussianModelRaDeGS
from gaussian_splatting.utils.sh_utils import eval_sh
from gsplat import rasterization


def render(viewpoint_camera,
           pc: GaussianModel,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           separate_sh=True,
           override_color=None,
           use_trained_exp=False) -> Dict[str, torch.Tensor]:
    """

    Parameters
    ----------
        viewpoint_camera
        pc
        pipe
        bg_color: Background tensor (bg_color) must be on GPU!
        scaling_modifier
        override_color
        return_depth:
            If True, a different GaussianRasterizer will be used that also returns depth.
            Note, that currently the depth rendering is not differentiable and does not include the speed improvements from distwar!

    Returns
    -------
        A dict with:
         - "render"
         - "viewspace_points"
         - "visibility_filter"
         - "radii"
         (- "depth": if return_depth=True)

    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    C = pc._features_dc.shape[2] if override_color is None else override_color.shape[1]
    NUM_CHANNELS = 32

    assert len(bg_color) == C, f"bg_color tensor must have same number of channels as feature tensor. Got {len(bg_color)} vs {C}"

    if 3 < C < NUM_CHANNELS:
        bg_color = torch.cat([bg_color, torch.zeros((NUM_CHANNELS - C,), dtype=bg_color.dtype, device=bg_color.device)],
                             dim=-1)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    if C > 3:
        # Use separate diff-gaussian-rasterization repo with N_CHANNELS=32 compiled CUDA
        rasterizer = GaussianFeatureRasterizer(raster_settings=raster_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        assert rotations.is_contiguous(), "rotations need to be contiguous to avoid rasterizer gradient bug!"

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python or C > 3:
            # if pipe.convert_SHs_python:  # Spherical Harmonics must always be computed in Python now, since rasterizer uses 32 channels
            shs_view = pc.get_features.transpose(1, 2).view(-1, C, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    if 3 < C < NUM_CHANNELS:
        G = colors_precomp.shape[0]
        colors_precomp = torch.cat([colors_precomp, torch.zeros((G, NUM_CHANNELS - C), device=colors_precomp.device,
                                                                dtype=colors_precomp.dtype)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    if C > 3:
        # If feature rasterizer was used, returned tensor has 32 channels. Only slice the actual used channels here
        rendered_image = rendered_image[:C]

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }

    return out

def render_distwar(viewpoint_camera,
           pc: GaussianModel,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           override_color=None):
    """

    Parameters
    ----------
        viewpoint_camera
        pc
        pipe
        bg_color: Background tensor (bg_color) must be on GPU!
        scaling_modifier
        override_color
        return_depth:
            If True, a different GaussianRasterizer will be used that also returns depth.
            Note, that currently the depth rendering is not differentiable and does not include the speed improvements from distwar!

    Returns
    -------
        A dict with:
         - "render"
         - "viewspace_points"
         - "visibility_filter"
         - "radii"
         (- "depth": if return_depth=True)

    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    C = pc._features_dc.shape[2] if override_color is None else override_color.shape[1]
    NUM_CHANNELS = 32

    assert len(bg_color) == C, f"bg_color tensor must have same number of channels as feature tensor. Got {len(bg_color)} vs {C}"

    if 3 < C < NUM_CHANNELS:
        bg_color = torch.cat([bg_color, torch.zeros((NUM_CHANNELS - C,), dtype=bg_color.dtype, device=bg_color.device)],
                             dim=-1)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    if C > 3:
        # Use separate diff-gaussian-rasterization repo with N_CHANNELS=32 compiled CUDA
        rasterizer = DistwarGaussianFeatureRasterizer(raster_settings=raster_settings)
    else:
        rasterizer = DistwarGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        assert rotations.is_contiguous(), "rotations need to be contiguous to avoid rasterizer gradient bug!"

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python or C > 3:  # Spherical Harmonics must always be computed in Python now, since rasterizer uses 32 channels
            shs_view = pc.get_features.transpose(1, 2).view(-1, C, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if 3 < C < NUM_CHANNELS:
        G = colors_precomp.shape[0]
        colors_precomp = torch.cat([colors_precomp, torch.zeros((G, NUM_CHANNELS - C), device=colors_precomp.device,
                                                                dtype=colors_precomp.dtype)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rasterizer_output = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_image, radii = rasterizer_output

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:C],
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def render_gsplat(viewpoint_camera,
                  pc: GaussianModel,
                  bg_color: torch.Tensor,
                  scaling_modifier=1.0,
                  override_color=None):
    """

    Parameters
    ----------
        viewpoint_camera
        pc
        bg_color: Background tensor (bg_color) must be on GPU!
        scaling_modifier
        override_color

    Returns
    -------
        A dict with:
         - "render"
         - "viewspace_points"
         - "visibility_filter"
         - "radii"
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.cx],
            [0, focal_length_y, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        device=pc._xyz.device,
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)  # [N,]
    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter": radii > 0,
            "radii": radii}


def render_radegs(viewpoint_camera,
                  pc: GaussianModelRaDeGS,
                  pipe,
                  bg_color: torch.Tensor,
                  kernel_size: float = 0.0,
                  scaling_modifier: float = 1.0,
                  override_color=None,
                  require_coord: bool = True,
                  require_depth: bool = True):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettingsRaDeGS(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerRaDeGS(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = override_color

    shs = None
    colors_precomp = None
    if override_color is None:
        C = pc._features_dc.shape[2]
        shs_view = pc.get_features.transpose(1, 2).view(-1, C, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "normal": rendered_normal,
            }
