from __future__ import annotations

import accelerated_image_processor.accelerated_image_processor_python_compression as compression_cpp

__all__ = [
    "create_compressor",
    "CompressionType",
]


class CompressionType(compression_cpp.CompressionType):
    """CompressionType is an enumeration that defines the compression types supported by the JPEGCompressor class."""


create_compressor = compression_cpp.create_compressor
create_compressor.__doc__ = (
    """create_compressor is a function that creates a JPEGCompressor object."""
)
