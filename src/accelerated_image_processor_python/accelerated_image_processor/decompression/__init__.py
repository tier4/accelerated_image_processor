from __future__ import annotations

import accelerated_image_processor.accelerated_image_processor_python_decompression as decompression_cpp

__all__ = ["DecompressionType", "create_decompressor"]


class DecompressionType(decompression_cpp.DecompressionType):
    """DecompressionType is an enumeration that defines the supported decompression types."""


create_decompressor = decompression_cpp.create_decompressor
create_decompressor.__doc__ = """Returns a concrete decompressor instance."""
