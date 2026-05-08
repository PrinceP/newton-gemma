"""Offscreen camera that grabs a single frame of a Newton model and writes a PNG.

Built on `newton.sensors.SensorTiledCamera` so it doesn't depend on OpenGL or
CUDA: it renders directly from the model's geometry on whichever Warp device
is available.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import warp as wp

from newton.sensors import SensorTiledCamera


_DEFAULT_FOV_RAD = float(np.deg2rad(45.0))


class SnapshotCamera:
    def __init__(
        self,
        model,
        *,
        width: int = 512,
        height: int = 384,
        fov_rad: float = _DEFAULT_FOV_RAD,
    ):
        self.model = model
        self.width = width
        self.height = height
        self.fov_rad = fov_rad
        self.world_count = max(1, getattr(model, "world_count", 1))

        self._sensor = SensorTiledCamera(model)
        self._rays = self._sensor.utils.compute_pinhole_camera_rays(width, height, [fov_rad])
        self._color = self._sensor.utils.create_color_image_output(width, height)

    @staticmethod
    def from_viewer_camera(viewer_camera) -> wp.transform:
        """Mirror the GL viewer's current camera as a camera-to-world wp.transform.

        SensorTiledCamera consumes camera-to-world transforms; the camera frame
        is OpenGL-style (camera looks down -Z, +Y up, +X right). Build the
        rotation from the camera's front/up/right basis directly — the GL
        view matrix is world-to-camera, which is the wrong direction.
        """
        pos = np.asarray(viewer_camera.pos, dtype=np.float32)
        front = np.asarray(viewer_camera.get_front(), dtype=np.float32)
        world_up = np.asarray(viewer_camera.get_up(), dtype=np.float32)
        right = np.cross(front, world_up)
        right /= max(np.linalg.norm(right), 1e-8)
        up = np.cross(right, front)
        # Columns: world-axes of camera +X, +Y, +Z
        rot = np.column_stack([right, up, -front]).astype(np.float32)
        mat33 = wp.mat33f(*rot.flatten().tolist())
        return wp.transformf(wp.vec3f(*pos.tolist()), wp.quat_from_matrix(mat33))

    def render(self, state, camera_transform: wp.transform | None = None) -> np.ndarray:
        """Render one frame and return an (H, W, 4) uint8 RGBA numpy array."""
        if camera_transform is None:
            camera_transform = wp.transformf(
                wp.vec3f(1.0, 0.0, 0.5), wp.quatf(0.5, 0.5, 0.5, 0.5)
            )

        cam_array = wp.array(
            [[camera_transform] * self.world_count],
            dtype=wp.transformf,
        )

        self._sensor.update(
            state,
            cam_array,
            self._rays,
            color_image=self._color,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )

        # Pull world 0, camera 0 out of the (W, C, H, W) buffer; bytes are R,G,B,A.
        u32 = self._color.numpy()[0, 0]  # (H, W) uint32
        return u32.view(np.uint8).reshape(self.height, self.width, 4)

    def save_snapshot(
        self,
        state,
        camera_transform: wp.transform | None = None,
        path: str | None = None,
    ) -> str:
        from PIL import Image  # noqa: PLC0415

        rgba = self.render(state, camera_transform)
        if path is None:
            tmpdir = tempfile.gettempdir()
            path = os.path.join(tmpdir, "newton_gemma_snapshot.png")
        Image.fromarray(rgba, mode="RGBA").save(path)
        return path


def screenshot_gl_viewer(viewer, path: str | None = None) -> str:
    """Read the GL viewer's current framebuffer pixels via plain glReadPixels
    (host-side, no CUDA-GL interop) and save as PNG. Must be called from the
    UI thread that owns the GL context — i.e. from a register_ui_callback or
    immediately after viewer.end_frame().
    """
    import ctypes  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415
    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    renderer = viewer.renderer
    w, h = renderer._screen_width, renderer._screen_height

    # Read from the rendered framebuffer (where the viewer's last frame
    # landed) -- not the default framebuffer, which may have ImGui on top.
    prev_fbo = (gl.GLint * 1)()
    gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING, prev_fbo)
    if renderer._frame_fbo is not None:
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, renderer._frame_fbo)

    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    buf = (ctypes.c_uint8 * (w * h * 3))()
    gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buf)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, prev_fbo[0])

    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    # OpenGL origin is bottom-left; PIL/standard images are top-left.
    arr = np.flipud(arr).copy()

    if path is None:
        path = os.path.join(tempfile.gettempdir(), "newton_gemma_screenshot.png")
    Image.fromarray(arr, mode="RGB").save(path)
    return path
