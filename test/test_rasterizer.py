import unittest

import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.matrix import Intrinsics, Pose
from dreifus.pyvista import add_camera_frustum
from dreifus.util.visualizer import ImageWindow
from dreifus.vector import Vec3
from tqdm import tqdm

from gaussian_splatting.arguments import PipelineParams2, OptimizationParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.graphics_utils import BasicPointCloud


import pyvista as pv

from gaussian_splatting.utils.loss_utils import l1_loss, ssim, fast_ssim
from gaussian_splatting.utils.viewer import GaussianViewer


class RasterizerTest(unittest.TestCase):

    def test_rasterizer(self):
        device = torch.device('cuda')
        C = 3
        sh_degree = 0
        points = torch.randn((5000, 3)) / 10
        colors = torch.randn((5000, 3))

        gaussian_model = GaussianModel(sh_degree)
        pointcloud = BasicPointCloud(points, colors, None)
        gaussian_model.create_from_pcd(pointcloud)
        gaussian_model._features_dc.data = torch.randn((5000, 1, C), device=device)
        gaussian_model._features_rest.data = torch.empty((5000, sh_degree * 3, C), device=device)
        gaussian_model.training_setup(OptimizationParams2(optimizer_type='sparse_adam'))

        intrinsics = Intrinsics(5000, 5000, 256, 256)
        pose = Pose(pose_type=PoseType.CAM_2_WORLD)
        pose.set_translation(z=5)
        pose.look_at(Vec3(), up=Vec3(0, 1, 0))

        # p = pv.Plotter()
        # p.add_points(points.detach().cpu().numpy())
        # add_camera_frustum(p, pose, intrinsics)
        # p.show()

        bg_color = torch.ones(C, device=device)

        target_img = torch.zeros((C, 512, 512), device=device)

        render_cam = pose_to_rendercam(pose, intrinsics, 512, 512)

        viewer = GaussianViewer(gaussian_model)
        image_buffer = np.zeros((512, 512, 3), dtype=np.float32)
        ImageWindow(image_buffer)

        progress = tqdm(range(10000))
        for i in progress:
            render_output = render(render_cam, gaussian_model, PipelineParams2(convert_SHs_python=True), bg_color)
            rendered_img = render_output['render']

            image_buffer[:] = rendered_img[:3].permute(1, 2, 0).detach().cpu().numpy()

            Ll1 = l1_loss(rendered_img, target_img)
            loss = (0.8) * Ll1 + 0.2 * (1.0 - fast_ssim(rendered_img, target_img))
            loss.backward()

            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad()
            gaussian_model._scaling.data[:] = -5  # Prevent scales from growing crazily

            progress.set_postfix(loss=loss.item())




        print('hi')