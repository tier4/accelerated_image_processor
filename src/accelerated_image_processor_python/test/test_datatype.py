from __future__ import annotations

from accelerated_image_processor.datatype import (
    CameraInfo,
    DistortionModel,
    Image,
    ImageEncoding,
    ImageFormat,
    Roi,
)


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

    data = []
    for y in range(height):
        for x in range(width):
            data.append(x % 256)
            data.append(y % 256)
            data.append((x + y) % 256)

    image.data = data


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
