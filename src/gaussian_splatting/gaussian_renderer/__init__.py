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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from depth_diff_gaussian_rasterization import GaussianRasterizer as GaussianDepthRasterizer
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

def render(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           return_depth: bool = False):
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

    C = pc._features_dc.shape[2]
    NUM_CHANNELS = 32

    if C < NUM_CHANNELS and not return_depth:
        bg_color = torch.cat([bg_color, torch.zeros((NUM_CHANNELS - C,), dtype=bg_color.dtype, device=bg_color.device)], dim=-1)

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
        debug=pipe.debug
    )

    if return_depth:
        rasterizer = GaussianDepthRasterizer(raster_settings=raster_settings)
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
    colors_precomp = None
    if override_color is None:
        if True:
        # if pipe.convert_SHs_python:  # Spherical Harmonics must always be computed in Python now, since rasterizer uses 32 channels
            shs_view = pc.get_features.transpose(1, 2).view(-1, C, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if C < NUM_CHANNELS and not return_depth:
        G = colors_precomp.shape[0]
        colors_precomp = torch.cat([colors_precomp, torch.zeros((G, NUM_CHANNELS - C), device=colors_precomp.device, dtype=colors_precomp.dtype)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rasterizer_output = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    if return_depth:
        rendered_image, radii, depth = rasterizer_output

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image[:C],
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": depth}
    else:
        rendered_image, radii = rasterizer_output

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image[:C],
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
