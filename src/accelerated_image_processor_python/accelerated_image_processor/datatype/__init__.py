from __future__ import annotations

import accelerated_image_processor.accelerated_image_processor_python_datatype as datatype_cpp

__all__ = [
    "ImageEncoding",
    "ImageFormat",
    "DistortionModel",
    "Image",
    "CameraInfo",
    "Roi",
]

# ------- Enums -------
ImageEncoding = datatype_cpp.ImageEncoding
ImageEncoding.__doc__ = """Enumeration of image color encodings."""

ImageFormat = datatype_cpp.ImageFormat
ImageFormat.__doc__ = """Enumeration of image compression formats."""

DistortionModel = datatype_cpp.DistortionModel
DistortionModel.__doc__ = """Enumeration of distortion models."""

# ------- Image -------
Image = datatype_cpp.Image
Image.__doc__ = """Class representing an image."""

CameraInfo = datatype_cpp.CameraInfo
CameraInfo.__doc__ = """Class representing camera information."""

Roi = datatype_cpp.Roi
Roi.__doc__ = """Class representing a region of interest."""
