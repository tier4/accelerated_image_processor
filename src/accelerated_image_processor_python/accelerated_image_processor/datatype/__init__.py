from __future__ import annotations

from os import PathLike

import accelerated_image_processor.accelerated_image_processor_python_datatype as datatype_cpp
import cv2
import numpy as np

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
class Image(datatype_cpp.Image):
    """Class representing an image."""

    @classmethod
    def from_numpy(cls, data: np.ndarray, encoding=ImageEncoding.RGB) -> Image:
        """Construct an image from a numpy array.

        Args:
            data (np.ndarray): The image data as a numpy array in the shape of (H, W, 3).
            encoding (ImageEncoding): The image color encoding.

        Returns:
            The constructed image.
        """
        if data.ndim != 3:
            raise ValueError(
                f"Expected ndim == 3 (H, W, 3), got ndim={data.ndim}, shape={data.shape}"
            )

        image = cls()
        image.encoding = encoding
        image.format = ImageFormat.RAW

        data_u8 = data.astype(np.uint8)

        height, width, channels = data_u8.shape
        if channels != 3:
            raise ValueError(f"Expected channels==3, got {channels}")
        image.height = int(height)
        image.width = int(width)
        image.step = int(width * channels)
        image.data = data_u8.ravel().tolist()

        return image

    @classmethod
    def from_file(cls, filepath: PathLike) -> Image:
        """Construct an image from a file.

        Args:
            filepath (PathLike): The path to the image file.

        Returns:
            The constructed image.
        """
        cv_image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError(f"Failed to read image: {filepath}")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cls.from_numpy(cv_image, ImageEncoding.RGB)


CameraInfo = datatype_cpp.CameraInfo
CameraInfo.__doc__ = """Class representing camera information."""

Roi = datatype_cpp.Roi
Roi.__doc__ = """Class representing a region of interest."""
