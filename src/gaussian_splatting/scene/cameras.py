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

from typing import Optional, Union

import numpy as np
import torch
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.matrix.intrinsics_numpy import fov_to_focal_length
from torch import nn

from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 cx: Optional[float] = None, cy: Optional[float] = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FoVx, fovY=self.FoVy,
                                                     width=self.image_width, height=self.image_height,
                                                     cx=self.cx, cy=self.cy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_pose(self) -> Pose:
        return GS_camera_to_pose(self)

    def to_intrinsics(self) -> Intrinsics:
        return GS_camera_to_intrinsics(self)

    @staticmethod
    def from_pose(pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int):
        return pose_to_GS_camera(pose, intrinsics, img_w, img_h)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class RenderCam:
    def __init__(self, width, height, R, T, FoVx, FoVy, cx: Optional[float] = None, cy: Optional[float] = None,
                 znear: float = 0.01, zfar: float = 100,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 device: torch.device = torch.device('cuda'),
                 ):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), device=device).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar,
                                                fovX=self.FoVx, fovY=self.FoVy,
                                                width=width, height=height,
                                                cx=cx, cy=cy).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_pose(self) -> Pose:
        return GS_camera_to_pose(self)

    def to_intrinsics(self) -> Intrinsics:
        return GS_camera_to_intrinsics(self)

    @staticmethod
    def from_pose(pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int, znear: float = 0.01, zfar: float = 100) -> 'RenderCam':
        return pose_to_rendercam(pose, intrinsics, img_w, img_h, znear=znear, zfar=zfar)


# ==========================================================
# Conversion between Gaussian Splatting camera and dreifus Pose
# ==========================================================


def GS_camera_to_pose(camera: Union[Camera, RenderCam]) -> Pose:
    pose = Pose(camera.R.transpose(), camera.T, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
    return pose


def GS_camera_to_intrinsics(camera: Union[Camera, RenderCam]) -> Intrinsics:
    fx = fov_to_focal_length(camera.FoVx, camera.image_width)
    fy = fov_to_focal_length(camera.FoVy, camera.image_height)
    cx = camera.cx
    cy = camera.cy
    intrinsics = Intrinsics(fx, fy, cx=cx, cy=cy)
    return intrinsics


def pose_to_GS_camera(pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int) -> Camera:
    fov_x = intrinsics.get_fovx(img_w)
    fov_y = intrinsics.get_fovy(img_h)
    cx = intrinsics.cx
    cy = intrinsics.cy
    dummy_img = torch.empty((3, img_h, img_w))  # TODO: Find another way, s.t. we do not have to allocate this all the time

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)
    T = pose.get_translation()
    R = pose.get_rotation_matrix().transpose()

    camera = Camera(0, R, T, fov_x, fov_y, dummy_img, None, None, None, cx=cx, cy=cy)

    return camera


def pose_to_rendercam(pose: Pose,
                      intrinsics: Intrinsics,
                      img_w: int,
                      img_h: int,
                      znear: float = 0.01,
                      zfar: float = 100,
                      device: torch.device = torch.device('cuda')) -> RenderCam:
    fov_x = intrinsics.get_fovx(img_w)
    fov_y = intrinsics.get_fovy(img_h)
    cx = intrinsics.cx
    cy = intrinsics.cy

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)
    T = pose.get_translation()
    R = pose.get_rotation_matrix().transpose()

    camera = RenderCam(img_w, img_h, R, T, fov_x, fov_y, cx=cx, cy=cy, znear=znear, zfar=zfar, device=device)

    return camera
