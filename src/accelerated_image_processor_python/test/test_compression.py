from __future__ import annotations

import numpy as np
import pytest
from accelerated_image_processor.common import Image, ImageFormat, fetch_parameters
from accelerated_image_processor.compression import CompressionType, create_compressor


def _make_image_array(height: int, width: int) -> np.ndarray:
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x, 0] = x % 256
            image_array[y, x, 1] = y % 256
            image_array[y, x, 2] = (x + y) % 256
    return image_array


@pytest.mark.parametrize("compression_type", [CompressionType.JPEG, "jpeg"])
def test_jpeg_compressor(compression_type: CompressionType):
    height, width = 1080, 1920
    image = Image.from_numpy(_make_image_array(height, width))

    compressor = create_compressor(compression_type)
    assert compressor is not None

    result = compressor.process(image)
    assert result is not None
    assert result.format == ImageFormat.JPEG


def test_fetch_parameters_for_jpeg_compressor():
    config = {"quality": 85}
    compressor = create_compressor(CompressionType.JPEG)

    processor = fetch_parameters(config, compressor)

    assert processor.parameters["quality"] == 85
