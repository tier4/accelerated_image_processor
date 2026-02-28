from __future__ import annotations

import accelerated_image_processor.accelerated_image_processor_python_compression as compression_cpp

__all__ = [
    "create_compressor",
    "CompressionBackend",
    "CompressionType",
]


class CompressionBackend(compression_cpp.CompressionBackend):
    """CompressionBackend is an enumeration that defines supported compression backends."""


class CompressionType(compression_cpp.CompressionType):
    """CompressionType is an enumeration that defines the supported compression types."""


create_compressor = compression_cpp.create_compressor
create_compressor.__doc__ = """Returns a concrete compressor instance."""
