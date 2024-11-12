# network_handler.py
import threading
import struct
import logging
import socket
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NetworkHandler(threading.Thread):
    """
    A class to handle network communication with a server.

    Attributes:
        host (str): The server host.
        port (int): The server port.
        sock (socket.socket): The socket used for communication.
        entity_id (Optional[int]): The ID of the entity.
        chunk_update_callback (Optional[Callable[[Tuple[int, int, int], np.ndarray], None]]): Callback function for chunk updates.
        running (bool): Flag to indicate if the handler is running.
    """
    def __init__(self, host, port):
        super().__init__()
        self.daemon = True
        self.running = True

        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.entity_id = None

        # callbacks
        self.chunk_update_callback = None

    def connect(self):
        self.sock.connect((self.host, self.port))
        logger.info("Connected to the server")

    def disconnect(self):
        self.sock.close()
        logger.info("Disconnected from the server")

    def send_packet(self, packet_id, data):
        self.sock.sendall(struct.pack("!B", packet_id) + data)

    def receive_packet(self):
        packet_id = struct.unpack("!B", self.recv_all(1))[0]
        return packet_id

    def recv_all(self, length):
        data = b""
        while len(data) < length:
            more_data = self.sock.recv(length - len(data))
            if not more_data:
                raise ConnectionError("Socket connection broken")
            data += more_data
        return data

    def handle_packet(self, packet_id):
        if packet_id == 0x00:
            self.handle_identification()
        elif packet_id == 0x01:
            self.handle_add_entity()
        elif packet_id == 0x02:
            self.handle_remove_entity()
        elif packet_id == 0x03:
            self.handle_update_entity()
        elif packet_id == 0x04:
            self.handle_receive_chunk()
        elif packet_id == 0x05:
            self.handle_receive_mono_type_chunk()
        elif packet_id == 0x06:
            self.handle_chat()
        elif packet_id == 0x07:
            self.handle_update_entity_metadata()
        else:
            logger.warning(f"Unknown packet ID: {packet_id}")

    def handle_identification(self):
        expected_length = 4
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            self.entity_id = struct.unpack("!I", data)[0]
            logger.info(f"Identification received, entity ID: {self.entity_id}")
        else:
            logger.error("Failed to receive the complete identification packet")

    def handle_add_entity(self):
        expected_length = 88  # 4 bytes for entityId, 20 bytes for floats, 64 bytes for name
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            entity_id, x, y, z, yaw, pitch, name_bytes = struct.unpack("!Ifffff64s", data)
            name = name_bytes.decode('utf-8').rstrip('\x00')
            logger.info(
                f"Add Entity: ID={entity_id}, X={x}, Y={y}, Z={z}, Yaw={yaw}, Pitch={pitch}, Name={name}"
            )
        else:
            logger.error("Failed to receive the complete add entity packet")

    def handle_chat(self):
        expected_length = 4096
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            message = data.decode('utf-8').rstrip('\x00')
            logger.info(f"Chat Message Received: {message}")
        else:
            logger.error("Failed to receive the complete chat packet")

    def handle_update_entity_metadata(self):
        expected_length = 68  # 4 bytes for entityId, 64 bytes for name
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            entity_id, name_bytes = struct.unpack("!I64s", data)
            name = name_bytes.decode('utf-8').rstrip('\x00')
            logger.info(f"Update Entity Metadata: ID={entity_id}, Name={name}")
        else:
            logger.error("Failed to receive the complete update entity metadata packet")

    def handle_remove_entity(self):
        expected_length = 4
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            entity_id = struct.unpack("!I", data)[0]
            logger.info(f"Remove Entity: ID={entity_id}")
        else:
            logger.error("Failed to receive the complete remove entity packet")

    def handle_update_entity(self):
        expected_length = 24
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            entity_id, x, y, z, yaw, pitch = struct.unpack("!Ifffff", data)
            logger.info(
                f"Update Entity: ID={entity_id}, X={x:.2f}, Y={y:.2f}, Z={z:.2f}, Yaw={yaw:.2f}, Pitch={pitch:.2f}"
            )
        else:
            logger.error("Failed to receive the complete update entity packet")

    def handle_receive_chunk(self):
        """
        Handles the reception of a chunk of data from the network.

        This method expects to receive a fixed-length packet of 4108 bytes,
        which consists of three integers (x, y, z) followed by 4096 bytes of block data.
        It unpacks the received data and logs the chunk coordinates.
        If a chunk update callback is set, it invokes the callback with the chunk coordinates
        and the block data as a numpy array.

        Raises:
            struct.error: If the received data cannot be unpacked correctly.

        Logs:
            Info: When a chunk is successfully received with its coordinates.
            Error: If the complete chunk packet is not received.

        """
        expected_length = 4108  # 3 ints + 4096 bytes = 12 + 4096 = 4108 bytes
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            x, y, z, block_data = struct.unpack("!iii4096s", data)
            logger.info(f"Received Chunk: X={x}, Y={y}, Z={z}")
            if self.chunk_update_callback is not None:
                self.chunk_update_callback(
                    (x, y, z), np.frombuffer(block_data, dtype=np.uint8)
                )
        else:
            logger.error("Failed to receive the complete chunk packet")

    def handle_receive_mono_type_chunk(self):
        """
        Handles the reception of a mono type chunk from the network.

        This method expects to receive a fixed-length packet of 13 bytes, which 
        consists of three integers (x, y, z) and one byte (block_type). It reads 
        the data from the network, unpacks it, and logs the received values. If 
        the received data length does not match the expected length, it logs an 
        error message.

        Expected packet structure:
        - 3 integers (4 bytes each) for x, y, z coordinates
        - 1 byte for block type

        Raises:
            struct.error: If unpacking the data fails due to incorrect format.
        """
        expected_length = 13  # 3 ints + 1 byte = 12 + 1 = 13 bytes
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            x, y, z, block_type = struct.unpack("!iiib", data)
            logger.info(
                f"Received Mono Type Chunk: X={x}, Y={y}, Z={z}, BlockType={block_type}"
            )
        else:
            logger.error("Failed to receive the complete mono type chunk packet")

    def send_update_entity(self, x, y, z, yaw, pitch):
        packet_id = 0x00
        data = struct.pack("!fffff", x, y, z, yaw, pitch)
        self.send_packet(packet_id, data)
        logger.info(
            f"Sent Update Entity: X={x}, Y={y}, Z={z}, Yaw={yaw}, Pitch={pitch}"
        )

    def send_update_block(self, block_type, x, y, z):
        packet_id = 0x01
        data = struct.pack("!Biii", block_type, x, y, z)
        self.send_packet(packet_id, data)
        logger.info(f"Sent Update Block: BlockType={block_type}, X={x}, Y={y}, Z={z}")

    def send_block_bulk_edit(self, blocks):
        packet_id = 0x02
        block_count = len(blocks)
        data = struct.pack("!I", block_count)
        for block in blocks:
            block_type, x, y, z = block
            data += struct.pack("!Biii", block_type, x, y, z)
        self.send_packet(packet_id, data)
        logger.info(f"Sent Block Bulk Edit: BlockCount={block_count}")

    def send_chat(self, message):
        packet_id = 0x03
        message_bytes = message.encode('utf-8')
        message_bytes = message_bytes.ljust(4096, b'\x00')
        data = message_bytes[:4096]
        self.send_packet(packet_id, data)
        logger.info(f"Sent Chat Message: {message}")

    def send_client_metadata(self, render_distance, name):
        packet_id = 0x04
        name_bytes = name.encode('utf-8')
        name_bytes = name_bytes.ljust(64, b'\x00')
        data = struct.pack("!B64s", render_distance, name_bytes)
        self.send_packet(packet_id, data)
        logger.info(f"Sent Client Metadata: RenderDistance={render_distance}, Name={name}")

    # callbacks
    def set_chunk_update_callback(self, ufunc):
        self.chunk_update_callback = ufunc

    def run(self):
        self.connect()
        try:
            while self.running:
                packet_id = self.receive_packet()
                self.handle_packet(packet_id)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()

    def stop(self):
        self.running = False
        # self.thread.join()


# Usage example
if __name__ == "__main__":
    network_handler = NetworkHandler("162.19.137.231", 15000)
    network_handler.start()

    import time

    time.sleep(10)
