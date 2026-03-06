from __future__ import annotations

import pytest
from accelerated_image_processor.decompression import DecompressionType, create_decompressor


@pytest.mark.parametrize("decompression_type", [DecompressionType.VIDEO, "video"])
def test_video_decompressor(decompression_type: DecompressionType | str):
    decompressor = create_decompressor(decompression_type)
    assert decompressor is not None
    decompressor.register_postprocess(
        lambda image: print(f"Processed image (H, W)={image.height}, {image.width}")
    )
