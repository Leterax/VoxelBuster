# game_client.py
import logging

LOGLEVEL = logging.INFO

# Configure logging for the entire script
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


import moderngl_window as mglw
import moderngl as mgl
from network.network_handler import NetworkHandler


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

    vsync = False
    log_level = LOGLEVEL

    num_chunks = 1
    chunk_size = 128

    samples = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera.set_position(31, 16, 32)
        self.camera.set_rotation(60, 12.5)

        # self.network_handler = NetworkHandler("162.19.137.231", 15000)
        # self.network_handler.set_chunk_update_callback(self.update_chunk)
        # self.network_handler.start()
        # # network event queue
        # self.network_queue = Queue(-1)


        # Prepare voxel data
        self.data = np.zeros((self.chunk_size, self.chunk_size, self.chunk_size), dtype=np.int32)
        # Populate the voxel data (e.g., create a cube in the center)
        self.data[4:12, 4:12, 4:12] = 1
        # create a floor
        self.data[:, 0, :] = 1

        # fill some random blocks
        for i in range(1000):
            x, y, z = np.random.randint(0, self.chunk_size, 3)
            self.data[x, y, z] = 1

        # Create the voxel buffer and upload the data
        self.block_data_buffer = self.ctx.buffer(self.data.astype('int32').tobytes())

        # Load shaders
        self.compute_shader = self.load_compute_shader("raymarch.glsl")
        self.quad_shader = self.load_program("quad.glsl")

        # Create a framebuffer with color and depth attachments
        self.raymarch_resolution = (1280, 720) 
        # self.raymarch_resolution = (1280 * 2, 720 * 2) 
        self.color_texture = self.ctx.texture(self.raymarch_resolution, 4, dtype="f4")
        self.depth_texture = self.ctx.depth_texture(self.raymarch_resolution)

        # Create a screen-aligned quad
        self.quad = quad_fs()

        self.ctx.finish()


    def update_chunk(self, chunk_pos, chunk_data):
        """Callback to update a chunk"""
        self.network_queue.put((chunk_pos, chunk_data))

    def render(self, time, frame_time):
        # log camera position and pitch/yaw
        # logger.info(f"Camera Position: {self.camera.position}")
        # logger.info(f"Camera Pitch: {self.camera.pitch}, Yaw: {self.camera.yaw}")

        # Bind the framebuffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Bind the voxel buffer to the correct binding point
        self.block_data_buffer.bind_to_storage_buffer(binding=0)

        # Bind the output image
        self.color_texture.bind_to_image(0, read=False, write=True)

        # Set uniforms for the compute shader
        self.compute_shader["u_view"].write(self.camera.matrix)
        self.compute_shader["u_proj"].write(self.camera.projection.matrix)
        self.compute_shader["u_cameraPos"].value = tuple(self.camera.position)
        self.compute_shader["u_voxelGridDim"].value = (self.chunk_size, self.chunk_size, self.chunk_size)
        self.compute_shader["u_screenSize"].value = self.raymarch_resolution

        # Dispatch compute shader
        group_x = (self.raymarch_resolution[0] + 15) // 16
        group_y = (self.raymarch_resolution[1] + 15) // 16
        self.compute_shader.run(group_x, group_y)

        # Render the quad with the raymarched image
        self.color_texture.use(location=0)
        self.quad.render(self.quad_shader)


if __name__ == "__main__":
    GameClient.run()
