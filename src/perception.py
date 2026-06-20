from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import cv2
import numpy as np
from ultralytics import YOLO
DEFAULT_GEROBACK_NAMES = (
    "gerobak",
    "ngalangin",
    "cart",
    "obstruction",
)
DEFAULT_SIDEWALK_NAMES = ("sidewalk",)
DEFAULT_ROAD_NAMES = ("road", "drivable_road")
DEFAULT_SIDEWALK_IDS = (1,)
DEFAULT_ROAD_IDS = (0,)
def _normalize_class(name: str) -> str:
    return name.strip().lower()
@dataclass
class MaskDetection:
    mask: np.ndarray
    score: float
    class_name: str
    polygon: np.ndarray | None = None
    class_id: int = 0
@dataclass
class DualModelOutput:
    gerobak: list[MaskDetection]
    sidewalk: list[MaskDetection]
    road: list[MaskDetection]
class YoloSegModel:
    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf = conf
        self.device = device
        self.model = YOLO(self.weights_path)
    def predict(self, frame: np.ndarray) -> list[MaskDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.masks is None or result.boxes is None:
            return []
        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = result.names
        polygons = result.masks.xy
        height, width = frame.shape[:2]
        detections: list[MaskDetection] = []
        for idx, class_id in enumerate(class_ids):
            class_name = names.get(class_id, str(class_id))
            mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            polygon = polygons[idx] if polygons is not None and idx < len(polygons) else None
            detections.append(
                MaskDetection(
                    mask=mask,
                    score=float(scores[idx]),
                    class_name=class_name,
                    polygon=polygon,
                    class_id=int(class_id),
                )
            )
        return detections
class DualModelPerception:
    def __init__(
        self,
        gerobak_weights: str | Path,
        env_weights: str | Path,
        gerobak_conf: float = 0.25,
        env_conf: float = 0.25,
        device: str | None = None,
        gerobak_class_names: Sequence[str] = DEFAULT_GEROBACK_NAMES,
        sidewalk_class_names: Sequence[str] = DEFAULT_SIDEWALK_NAMES,
        road_class_names: Sequence[str] = DEFAULT_ROAD_NAMES,
        sidewalk_class_ids: Sequence[int] = DEFAULT_SIDEWALK_IDS,
        road_class_ids: Sequence[int] = DEFAULT_ROAD_IDS,
    ) -> None:
        self.gerobak_model = YoloSegModel(gerobak_weights, conf=gerobak_conf, device=device)
        self.env_model = YoloSegModel(env_weights, conf=env_conf, device=device)
        self.gerobak_names = {_normalize_class(name) for name in gerobak_class_names}
        self.sidewalk_names = {_normalize_class(name) for name in sidewalk_class_names}
        self.road_names = {_normalize_class(name) for name in road_class_names}
        self.sidewalk_ids = {int(idx) for idx in sidewalk_class_ids}
        self.road_ids = {int(idx) for idx in road_class_ids}
    def _matches(self, detection: MaskDetection, names: set[str], ids: set[int]) -> bool:
        name = _normalize_class(detection.class_name)
        if names and name in names:
            return True
        if ids and detection.class_id in ids:
            return True
        return False
    def predict(self, frame: np.ndarray) -> DualModelOutput:
        gerobak = self.gerobak_model.predict(frame)
        env_detections = self.env_model.predict(frame)
        sidewalk = []
        road = []
        for det in env_detections:
            if self._matches(det, self.sidewalk_names, self.sidewalk_ids):
                sidewalk.append(det)
            elif self._matches(det, self.road_names, self.road_ids):
                road.append(det)
            else:
                gerobak.append(det)
        return DualModelOutput(gerobak=gerobak, sidewalk=sidewalk, road=road)
class PerceptionModel:
    def __init__(
        self,
        weights_path: str | Path,
        target_class: str | list[str] | None = "NGALANGIN",
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.target_class = target_class
        self.conf = conf
        self.device = device
        self.model = YOLO(self.weights_path)
    def predict(self, frame: np.ndarray) -> list[MaskDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.masks is None or result.boxes is None:
            return []
        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = result.names
        polygons = result.masks.xy
        height, width = frame.shape[:2]
        detections: list[MaskDetection] = []
        for idx, class_id in enumerate(class_ids):
            class_name = names.get(class_id, str(class_id))
            if self.target_class is not None:
                if isinstance(self.target_class, str):
                    if self.target_class != "ALL" and class_name != self.target_class:
                        continue
                elif isinstance(self.target_class, (list, tuple, set)):
                    if class_name not in self.target_class:
                        continue
            mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            polygon = polygons[idx] if polygons is not None and idx < len(polygons) else None
            detections.append(
                MaskDetection(
                    mask=mask,
                    score=float(scores[idx]),
                    class_name=class_name,
                    polygon=polygon,
                    class_id=int(class_id),
                )
            )
        return detections
def combine_masks(detections: Iterable[MaskDetection], frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    combined = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        mask = detection.mask
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, mask)
    return combined
