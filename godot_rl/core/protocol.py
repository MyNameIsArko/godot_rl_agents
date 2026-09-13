"""Length-prefixed JSON transport for the Godot RL protocol."""

import json
import math
import numbers
import struct


MAX_FRAME_SIZE = 16 * 1024 * 1024
CONNECTION_TIMEOUT = 30
READ_TIMEOUT = 60
_LENGTH = struct.Struct("<I")


class ProtocolError(ValueError):
    """Raised when the peer sends an invalid protocol frame."""


def validate_finite(value, path="value"):
    """Reject NaN and infinity anywhere in a JSON value."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            raise ProtocolError(f"{path} contains NaN or infinity")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite(item, f"{path}[{index}]")


def encode_frame(message):
    validate_finite(message)
    try:
        body = json.dumps(message, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not valid JSON: {exc}") from exc
    if len(body) > MAX_FRAME_SIZE:
        raise ProtocolError(f"frame is {len(body)} bytes, maximum is {MAX_FRAME_SIZE}")
    return _LENGTH.pack(len(body)) + body


def _recv_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ProtocolError("peer closed during frame")
        data.extend(chunk)
    return bytes(data)


def recv_frame(connection):
    size = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))[0]
    if size > MAX_FRAME_SIZE:
        raise ProtocolError(f"frame is {size} bytes, maximum is {MAX_FRAME_SIZE}")
    try:
        message = json.loads(_recv_exact(connection, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"frame body is not valid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("frame JSON must be an object")
    validate_finite(message)
    return message


def send_frame(connection, message):
    connection.sendall(encode_frame(message))
