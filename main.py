# game_client.py
import logging

LOGLEVEL = logging.WARN

# Configure logging for the entire script
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


import moderngl_window as mglw
import moderngl as mgl
from network_handler import NetworkHandler

from camera_window import CameraWindow
import numpy as np
from pathlib import Path
from queue import Queue
from moderngl_window.geometry import quad_fs
import math


def normalize(v):
    return v / np.linalg.norm(v)


class GameClient(CameraWindow):
    title = "Game Client"
    window_size = (1280, 720)
    vsync = False
    gl_version = (4, 3)
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    vsync = True
    log_level = LOGLEVEL


    near_far_planes = (0.0, 2**16)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.network_handler = NetworkHandler("162.19.137.231", 15000)
        self.network_handler.set_chunk_update_callback(self.update_chunk)
        self.network_handler.start()
        # network event queue
        self.network_queue = Queue(-1)

        # Initialize chunk buffer
        self.chunk_data_buffer = self.ctx.buffer(
            b"\x00" * (1024 * 1024 * 1024)
        )

        # Load shaders
        self.compute_shader = self.load_compute_shader("raymarching.glsl")
        self.quad_shader = self.load_program("quad.glsl")

        # Create a framebuffer with color and depth attachments
        self.color_texture = self.ctx.texture(self.window_size, 4, dtype="f4")
        self.depth_texture = self.ctx.depth_texture(self.window_size)

        # Create a screen-aligned quad
        self.quad = quad_fs()



        self._camera = self.ctx.buffer(
            self.camera_creation(
                self.camera.position, self.camera.position + self.camera.dir
            )
        )

        # fill the buffer with some data (plane at y=0)
        self.data = np.zeros((1024,1024,1024), dtype=np.uint8)
        # set a 16x16x16 cube at 0,0,0
        self.data[0:16, 0:16, 0:16] = np.random.randint(0, 7, (16, 16, 16))

        # self.camera.set_position(512, 32, 512)


        self.chunk_data_buffer.write(self.data.ravel())

    def update_chunk(self, chunk_pos, chunk_data):
        """Callback to update a chunk"""
        self.network_queue.put((chunk_pos, chunk_data))

    def render(self, time, frame_time):
        # move camera
        self.camera.matrix
        if self.camera_enabled:
            self._camera.write(
                self.camera_creation(
                    self.camera.position, self.camera.position + self.camera.dir
                    )
            )


        # print(f"Camera position: {self.camera.position}")

        # get updates form network
        # while not self.network_queue.empty():
        #     position, chunk_data = self.network_queue.get_nowait()
        #     c_x, c_y, c_z = position
        #     c_x, c_y, c_z = c_x + 64, c_y + 32, c_z + 64
        #     chunk_size = 16
        #     # Write the chunk data to the buffer
        #     chunk = np.frombuffer(
        #         chunk_data, dtype=np.uint8
        #     ).reshape((chunk_size, chunk_size, chunk_size))

        #     # Swap the axes to match the expected layout
        #     chunk = np.swapaxes(chunk, 1, 2)



        #     print(f"Position {position}, Chunk size: {chunk.shape}, writing to data[{c_x}:{c_x+chunk_size},{c_y}:{c_y+chunk_size},{c_z}:{c_z+chunk_size}]")
        #     self.data[c_x : c_x + chunk_size, c_y : c_y + chunk_size, c_z : c_z + chunk_size] = chunk

        #     logging.log(
        #         logging.INFO,
        #         f"Writing {len(chunk_data)} bytes to buffer.",
        #     )
        #     # Update the buffer with the new data (only the chunk)
        #     self.chunk_data_buffer.write(
        #         self.data.ravel()
        #     )


        # Bind the framebuffer

        # Clear the screen
        # self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Compute shader dispatch
        self.chunk_data_buffer.bind_to_storage_buffer(0)

        # Bind the output texture and depth texture
        self.color_texture.bind_to_image(1, read=False, write=True)
        self.depth_texture.bind_to_image(2, read=False, write=True)

        self._camera.bind_to_uniform_block(binding=3)

        self.compute_shader.run(
            group_x=self.window_size[0] // 32, group_y=self.window_size[1] // 32
        )

        # Blit the result to the default framebuffer
        self.color_texture.use(0)
        self.quad.render(self.quad_shader)

    @staticmethod
    def camera_creation(eye, center=(0, 0, 0), up=(0, 1, 0), fov=45, aspect=16 / 9, aperture=0.01, focus_distance=1.0):
        lens_radius = aperture / 2
        theta = fov * math.pi / 180
        half_height = math.tan(theta / 2.0)
        half_width = aspect * half_height

        origin = eye
        w = normalize(eye - center)
        u = normalize(np.cross(up, w))
        v = np.cross(w, u)

        lower_left_corner = (
            origin - half_width * u * focus_distance - half_height * v * focus_distance - w * focus_distance
        )
        horizontal = 2 * half_width * u * focus_distance
        vertical = 2 * half_height * v * focus_distance

        padding = np.array([0.0])

        return np.concatenate(
            (
                eye,
                padding,
                lower_left_corner,
                padding,
                horizontal,
                padding,
                vertical,
                padding,
                origin,
                padding,
                u,
                padding,
                v,
                padding,
                w,
                padding,
                np.array([lens_radius]),
            ),
            axis=0,
        ).astype("f4")



if __name__ == "__main__":
    GameClient.run()
