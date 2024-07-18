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
        expected_length = 24
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            entity_id, x, y, z, yaw, pitch = struct.unpack("!Ifffff", data)
            logger.info(
                f"Add Entity: ID={entity_id}, X={x}, Y={y}, Z={z}, Yaw={yaw}, Pitch={pitch}"
            )
        else:
            logger.error("Failed to receive the complete add entity packet")

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
        expected_length = 13  # 3 ints + 1 byte = 12 + 1 = 13 bytes
        data = self.recv_all(expected_length)
        if len(data) == expected_length:
            x, y, z, block_type = struct.unpack("!iiib", data)
            logger.info(
                f"Received Mono Type Chunk: X={x}, Y={y}, Z={z}, BlockType={block_type}"
            )
        else:
            logger.error("Failed to receive the complete mono type chunk packet")

    def send_update_entity(self, entity_id, x, y, z, yaw, pitch):
        packet_id = 0x00
        data = struct.pack("!Ifffff", entity_id, x, y, z, yaw, pitch)
        self.send_packet(packet_id, data)
        logger.info(
            f"Sent Update Entity: ID={entity_id}, X={x}, Y={y}, Z={z}, Yaw={yaw}, Pitch={pitch}"
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
