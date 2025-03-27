import time
from abc import abstractmethod
from collections import deque
from threading import Thread
from typing import Tuple, List, Dict

import numpy as np
import torch
import viser
import viser.transforms
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.matrix.intrinsics_numpy import fov_to_focal_length
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_GS_camera
from viser import _messages
from viser.transforms import SO3


class ViserViewer:
    def __init__(self, viewer_port: int = 8008, poses: List[Pose] = None, intrinsics: List[Intrinsics] = None, images: List[np.ndarray] = None, size: float = 0.3):
        # self.device = device
        self.port = viewer_port

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button("Reset View")

        self.need_update = False

        self.pause_training = False

        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Train Viewer Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        if poses is not None:
            self.add_camera_frustums(poses, intrinsics, images, size=size)

        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = viser.transforms.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            # Initialize position to look at center of gaussians
            center = self._gaussian_model.get_xyz.mean(axis=0).detach().cpu().numpy()
            client.camera.position = np.array(center + np.array([0, 0, 1]))
            client.camera.look_at = center

            client.resolution_display = client.add_gui_text("Resolution", initial_value="0x0", disabled=True)

            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

        thread = Thread(target=self._worker, daemon=True)
        thread.start()

    def add_camera_frustums(self, poses: List[Pose], intrinsics: List[Intrinsics], images: List[np.ndarray], size: float = 0.3) -> None:
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}

        for cam_id, (pose, intr, image) in enumerate(zip(poses, intrinsics, images)):
            cam_to_world_pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torch.from_numpy(image).permute(2, 0, 1)
            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()

            R = SO3.from_matrix(cam_to_world_pose.get_rotation_matrix())
            fov_x = intr.get_fovx(image.shape[1])
            aspect = image.shape[1] / image.shape[0]

            name = f"/cameras/camera_{cam_id:05d}"
            camera_handle = self.server.add_camera_frustum(
                name=name,
                fov=fov_x,
                scale=size,
                aspect=aspect,
                image=image_uint8,
                wxyz=R.wxyz,
                position=cam_to_world_pose.get_translation(),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.server._websock_server.queue_message(
                _messages.SetOrientationMessage(
                    name=name,
                    wxyz=tuple(R.wxyz)
                )
            )

            self.server._websock_server.queue_message(
                _messages.SetPositionMessage(
                    name=name,
                    position=tuple(cam_to_world_pose.get_translation())
                )
            )

            self.camera_handles[cam_id] = camera_handle
            self.original_c2w[cam_id] = cam_to_world_pose


    def _worker(self):
        while True:
            time.sleep(0.1)
            self.update()

    @abstractmethod
    def _render(self, cam_to_world_pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @torch.no_grad()
    def update(self):
        # TODO: Don't run update loop if no client is connected
        if self.need_update:
            times = []
            for client in self.server.get_clients().values():
                camera = client.camera

                cam_to_world = viser.transforms.SE3.from_rotation_and_translation(viser.transforms.SO3(camera.wxyz), camera.position).as_matrix()
                if np.isnan(cam_to_world).any():
                    continue
                cam_to_world_pose = Pose(cam_to_world, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.CAM_2_WORLD)

                img_h = self.resolution_slider.value
                img_w = int(camera.aspect * img_h)
                client.resolution_display.value = f"{img_w}x{img_h}"

                fx = fov_to_focal_length(camera.fov, img_h)
                fy = fov_to_focal_length(camera.fov, img_h)

                intrinsics = Intrinsics(fx, fy, img_w / 2, img_h / 2)

                try:
                    start = time.time()

                    out, depth = self._render(cam_to_world_pose, intrinsics, img_w, img_h)

                    end = time.time()
                    times.append(end - start)
                except RuntimeError as e:
                    print(e)
                    continue
                client.set_background_image(out, format="jpeg", depth=depth)

                del out

            if len(times) > 0:
                self.render_times.append(np.mean(times))
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"

            # print(f"Update time: {end - start:.3g}")


class GaussianViewer(ViserViewer):

    def __init__(self,
                 gaussian_model: GaussianModel,
                 viewer_port: int = 8008,
                 background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 poses: List[Pose] = None, intrinsics: List[Intrinsics] = None, images: List[np.ndarray] = None, size: float = 0.3):
        self._background_color = torch.Tensor(background_color).cuda()
        self._gaussian_model = gaussian_model

        super(GaussianViewer, self).__init__(viewer_port=viewer_port, poses=poses, intrinsics=intrinsics, images=images, size=size)

        self.sh_order = self.server.add_gui_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )

    def _render(self, cam_to_world_pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        gs_camera = pose_to_GS_camera(cam_to_world_pose, intrinsics, img_w, img_h)

        render_result = render(gs_camera, self._gaussian_model, PipelineParams2(), self._background_color, return_depth=True)
        rendered_image = render_result['render']
        rendered_depth = render_result['depth']
        color = (rendered_image.permute(1, 2, 0).detach().cpu() * 255).clamp(0, 255).numpy().astype(np.uint8)
        depth = rendered_depth.permute(1, 2, 0).detach().cpu().numpy()
        depth[depth == 0] = np.inf

        return color, depth
