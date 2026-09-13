import json
import socket
import struct
import threading

import pytest

from godot_rl.core.protocol import MAX_FRAME_SIZE, ProtocolError, encode_frame, recv_frame


def test_frame_handles_fragmented_reads():
    sender, receiver = socket.socketpair()
    frame = encode_frame({"type": "reset", "value": "ok"})

    def send_one_byte_at_a_time():
        for byte in frame:
            sender.send(bytes([byte]))
        sender.close()

    thread = threading.Thread(target=send_one_byte_at_a_time)
    thread.start()
    assert recv_frame(receiver) == {"type": "reset", "value": "ok"}
    thread.join(timeout=2)
    receiver.close()


def test_frame_size_limit_is_checked_before_body_read():
    sender, receiver = socket.socketpair()
    sender.sendall(struct.pack("<I", MAX_FRAME_SIZE + 1))
    with pytest.raises(ProtocolError, match="maximum"):
        recv_frame(receiver)
    sender.close()
    receiver.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value):
    with pytest.raises(ProtocolError, match="NaN or infinity"):
        encode_frame({"value": value})


def test_partial_frame_is_rejected():
    sender, receiver = socket.socketpair()
    sender.sendall(struct.pack("<I", 4) + b"{}")
    sender.close()
    with pytest.raises(ProtocolError, match="closed during frame"):
        recv_frame(receiver)
    receiver.close()


def test_frame_body_must_be_a_json_object():
    sender, receiver = socket.socketpair()
    body = json.dumps(["not", "an", "object"]).encode()
    sender.sendall(struct.pack("<I", len(body)) + body)
    with pytest.raises(ProtocolError, match="must be an object"):
        recv_frame(receiver)
    sender.close()
    receiver.close()
