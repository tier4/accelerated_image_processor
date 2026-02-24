from __future__ import annotations

from os import PathLike
from typing import Any

import accelerated_image_processor.accelerated_image_processor_python_common as common_cpp
import cv2
import numpy as np

__all__ = [
    "is_jetson_available",
    "is_nvjpeg_available",
    "is_turbojpeg_available",
    "ImageEncoding",
    "ImageFormat",
    "DistortionModel",
    "Image",
    "CameraInfo",
    "Roi",
    "ParameterMap",
    "fetch_parameters",
]

# ------- Backends -------
is_jetson_available = common_cpp.is_jetson_available
is_jetson_available.__doc__ = "Return True if Jetson backend is available."

is_nvjpeg_available = common_cpp.is_nvjpeg_available
is_nvjpeg_available.__doc__ = "Return True if NVJPEG backend is available."

is_turbojpeg_available = common_cpp.is_turbojpeg_available
is_turbojpeg_available.__doc__ = "Return True if TurboJPEG backend is available."


# ------- Enums -------
class ImageEncoding(common_cpp.ImageEncoding):
    """Enumeration of image color encodings."""


class ImageFormat(common_cpp.ImageFormat):
    """Enumeration of image compression formats."""


class DistortionModel(common_cpp.DistortionModel):
    """Enumeration of distortion models."""


# ------- Image -------
class Image(common_cpp.Image):
    """Class representing an image."""

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        encoding: ImageEncoding = ImageEncoding.RGB,
    ) -> Image:
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


class CameraInfo(common_cpp.CameraInfo):
    """Class representing camera information."""


class Roi(common_cpp.Roi):
    """Class representing a region of interest."""


class ParameterMap(common_cpp.ParameterMap):
    """Class representing a map of parameters."""


class BaseProcessor(common_cpp.BaseProcessor):
    """Base class for processors."""


def fetch_parameters(config: dict[str, Any], processor: BaseProcessor) -> BaseProcessor:
    """Fetch parameters from a configuration dictionary and set them on the processor.

    Args:
        config (dict[str, Any]): The configuration dictionary.
        processor (BaseProcessor): The processor to set the parameters on.

    Returns:
        The processor with the parameters set.
    """
    # NOTE: processor.parameters cannot be modified directly
    params = processor.parameters
    for key, value in config.items():
        if key in params:
            params[key] = value
    processor.parameters = params

    return processor
