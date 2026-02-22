from __future__ import annotations

import cv2
import numpy as np
from accelerated_image_processor.datatype import (
    CameraInfo,
    DistortionModel,
    Image,
    ImageEncoding,
    ImageFormat,
    Roi,
)


def _make_image_array(height: int, width: int) -> np.ndarray:
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x, 0] = x % 256
            image_array[y, x, 1] = y % 256
            image_array[y, x, 2] = (x + y) % 256
    return image_array


def test_image_initialization():
    height, width = 1080, 1920

    image = Image()
    image.frame_id = "camera"
    image.timestamp = 1234567890
    image.height = height
    image.width = width
    image.step = width * 3
    image.encoding = ImageEncoding.RGB
    image.format = ImageFormat.RAW
    image.data = _make_image_array(height, width).ravel().tolist()


def test_image_from_numpy():
    height, width = 1080, 1920

    image_array = _make_image_array(height, width)

    image = Image.from_numpy(image_array)
    assert image.height == height
    assert image.width == width
    assert image.step == width * 3
    assert image.encoding == ImageEncoding.RGB
    assert image.format == ImageFormat.RAW
    assert len(image.data) == height * width * 3


def test_image_from_file(tmp_path):
    height, width = 1080, 1920

    image_array = _make_image_array(height, width)
    cv2.imwrite(str(tmp_path / "image.png"), image_array)

    image = Image.from_file(tmp_path / "image.png")
    assert image.height == height
    assert image.width == width
    assert image.step == width * 3
    assert image.encoding == ImageEncoding.RGB
    assert image.format == ImageFormat.RAW
    assert len(image.data) == height * width * 3


def test_camera_info_initialization():
    height, width = 1080, 1920

    roi = Roi()
    roi.x_offset = 0
    roi.y_offset = 0
    roi.width = width
    roi.height = height

    info = CameraInfo()
    info.frame_id = "camera"
    info.timestamp = 1234567890
    info.height = height
    info.width = width
    info.distortion_model = DistortionModel.PLUMB_BOB
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [1.0, 0.0, width / 2.0, 0.0, 1.0, height / 2.0, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [
        1.0,
        0.0,
        width / 2.0,
        0.0,
        0.0,
        1.0,
        height / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    info.binning_x = 1
    info.binning_y = 1
    info.roi = roi
