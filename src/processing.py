from __future__ import annotations
from typing import Iterable
import cv2
import numpy as np
def clean_mask(mask: np.ndarray, kernel_size: int=3, iterations: int=1) -> np.ndarray:
    if mask is None:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
def clean_masks(masks: Iterable[np.ndarray], kernel_size: int=3, iterations: int=1) -> list[np.ndarray]:
    return [clean_mask(mask, kernel_size=kernel_size, iterations=iterations) for mask in masks]
def refine_mask(mask: np.ndarray) -> np.ndarray:
    return clean_mask(mask)
