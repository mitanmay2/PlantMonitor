"""Image loading and preprocessing helpers."""

from __future__ import annotations

from typing import BinaryIO, Tuple

import cv2
import numpy as np

IMAGE_SIZE: Tuple[int, int] = (256, 256)


def decode_uploaded_image(uploaded_file: BinaryIO) -> np.ndarray:
    """Decode a Streamlit uploaded file into an OpenCV BGR image."""
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid JPG, JPEG, or PNG image.")
    return image


def preprocess_image(
    image_bgr: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize an image and return BGR, RGB, and HSV representations."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot process an empty image.")

    resized_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    return resized_bgr, image_rgb, image_hsv
