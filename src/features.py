from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


# detect fast keypoints
def detect_fast_keypoints(gray_frame: np.ndarray, mask: np.ndarray, threshold: int=25, nonmax_suppression: bool=True) -> list[cv2.KeyPoint]:
    if mask is None or mask.size == 0:
        return []

    if mask.shape[:2] != gray_frame.shape[:2]:
        mask = cv2.resize(mask, (gray_frame.shape[1], gray_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
    return detector.detect(gray_frame, mask=mask)

# fast density
def fast_density(keypoints: Iterable[cv2.KeyPoint], mask: np.ndarray) -> float:
    area = max(int(cv2.countNonZero(mask)), 1)
    return float(len(list(keypoints))) / float(area)

# is false positive
def is_false_positive(keypoints: Iterable[cv2.KeyPoint], mask: np.ndarray, min_density: float, min_count: int=0) -> bool:
    count = len(list(keypoints))

    if count < min_count:
        return True

    if fast_density(keypoints, mask) < min_density:
        return True
    return False

# verify mask with fast
def verify_mask_with_fast(gray_frame: np.ndarray, mask: np.ndarray, fast_threshold: int, min_density: float, min_count: int) -> tuple[list[cv2.KeyPoint], bool]:
    keypoints = detect_fast_keypoints(gray_frame, mask, threshold=fast_threshold)
    return (keypoints, not is_false_positive(keypoints, mask, min_density, min_count))

# verify gerobak mask
def verify_gerobak_mask(gray_frame: np.ndarray, mask: np.ndarray, threshold: int=25, min_keypoints: int=12) -> bool:
    keypoints = detect_fast_keypoints(gray_frame, mask, threshold=threshold)
    return len(keypoints) >= min_keypoints

# draw keypoints
def draw_keypoints(frame: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:

    if frame is None or frame.size == 0:
        return frame

    if not keypoints:
        return frame
    return cv2.drawKeypoints(frame, keypoints, frame.copy(), color=(0, 255, 255), flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
