from __future__ import annotations
from typing import Iterable
import cv2
import numpy as np
def compute_homography(src_points: Iterable[Iterable[float]], dst_points: Iterable[Iterable[float]]) -> np.ndarray:
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError('src_points and dst_points must be 4x2 lists')
    return cv2.getPerspectiveTransform(src, dst)
def warp_mask(mask: np.ndarray, homography: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(mask, homography, dst_size, flags=cv2.INTER_NEAREST)
def pixel_to_meter_ratio(pixel_width: float, real_width_m: float) -> float:
    if pixel_width <= 0 or real_width_m <= 0:
        raise ValueError('pixel_width and real_width_m must be positive')
    return float(real_width_m) / float(pixel_width)
def compute_area_m2(bev_mask: np.ndarray, pixel_to_meter: float) -> float:
    nonzero = cv2.countNonZero(bev_mask)
    return float(nonzero) * pixel_to_meter ** 2
def compute_accessibility(clear_area_m2: float, total_area_m2: float) -> float:
    if total_area_m2 <= 0:
        return 0.0
    return float(clear_area_m2) / float(total_area_m2) * 100.0
