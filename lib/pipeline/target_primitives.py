from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

TARGET_SCOPE_ASSET = "asset"
TARGET_SCOPE_FRAME = "frame"
TARGET_SCOPE_REGION = "region"
VALID_TARGET_SCOPES = {TARGET_SCOPE_ASSET, TARGET_SCOPE_FRAME, TARGET_SCOPE_REGION}


@dataclass
class InferenceTarget:
    target_id: str
    scope: str
    source_asset_id: str
    frame_index: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    parent_target_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.scope not in VALID_TARGET_SCOPES:
            raise ValueError(f"Invalid target scope: {self.scope}")
        if not self.target_id:
            raise ValueError("target_id is required")
        if not self.source_asset_id:
            raise ValueError("source_asset_id is required")
        if self.scope == TARGET_SCOPE_REGION and self.bbox is None:
            raise ValueError("bbox is required for region targets")
        if self.bbox is not None:
            _validate_bbox(self.bbox)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "target_id": self.target_id,
            "scope": self.scope,
            "source_asset_id": self.source_asset_id,
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "parent_target_id": self.parent_target_id,
            "metadata": self.metadata,
        }
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class BBoxValidationResult:
    bbox: Optional[Tuple[float, float, float, float]]
    valid: bool
    error: Optional[str] = None


def _validate_bbox(bbox: Sequence[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 numeric values: [x1, y1, x2, y2]")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")


def sanitize_bbox(
    bbox: Sequence[float],
    source_width: Optional[float] = None,
    source_height: Optional[float] = None,
) -> BBoxValidationResult:
    if bbox is None:
        return BBoxValidationResult(bbox=None, valid=False, error="bbox is required")

    if len(bbox) != 4:
        return BBoxValidationResult(
            bbox=None,
            valid=False,
            error="bbox must contain exactly 4 numeric values: [x1, y1, x2, y2]",
        )

    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return BBoxValidationResult(
            bbox=None,
            valid=False,
            error="bbox values must be numeric",
        )

    if source_width is not None:
        source_width = float(source_width)
        x1 = min(max(x1, 0.0), source_width)
        x2 = min(max(x2, 0.0), source_width)

    if source_height is not None:
        source_height = float(source_height)
        y1 = min(max(y1, 0.0), source_height)
        y2 = min(max(y2, 0.0), source_height)

    sanitized_bbox = (x1, y1, x2, y2)
    try:
        _validate_bbox(sanitized_bbox)
    except ValueError as exc:
        return BBoxValidationResult(bbox=None, valid=False, error=str(exc))

    return BBoxValidationResult(bbox=sanitized_bbox, valid=True)


def build_asset_target(
    source_asset_id: str,
    target_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> InferenceTarget:
    target = InferenceTarget(
        target_id=target_id or _build_target_id(TARGET_SCOPE_ASSET),
        scope=TARGET_SCOPE_ASSET,
        source_asset_id=source_asset_id,
        metadata=metadata or {},
    )
    target.validate()
    return target


def build_frame_target(
    source_asset_id: str,
    frame_index: float,
    parent_target_id: Optional[str] = None,
    target_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> InferenceTarget:
    target = InferenceTarget(
        target_id=target_id or _build_target_id(TARGET_SCOPE_FRAME),
        scope=TARGET_SCOPE_FRAME,
        source_asset_id=source_asset_id,
        frame_index=float(frame_index),
        parent_target_id=parent_target_id,
        metadata=metadata or {},
    )
    target.validate()
    return target


def build_region_target(
    source_asset_id: str,
    bbox: Sequence[float],
    frame_index: Optional[float] = None,
    parent_target_id: Optional[str] = None,
    source_width: Optional[float] = None,
    source_height: Optional[float] = None,
    target_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> InferenceTarget:
    sanitized = sanitize_bbox(bbox, source_width=source_width, source_height=source_height)
    if not sanitized.valid:
        raise ValueError(sanitized.error)

    target = InferenceTarget(
        target_id=target_id or _build_target_id(TARGET_SCOPE_REGION),
        scope=TARGET_SCOPE_REGION,
        source_asset_id=source_asset_id,
        frame_index=float(frame_index) if frame_index is not None else None,
        bbox=sanitized.bbox,
        parent_target_id=parent_target_id,
        metadata=metadata or {},
    )
    target.validate()
    return target


def image_paths_to_asset_targets(image_paths: Iterable[str]) -> List[InferenceTarget]:
    targets = []
    for index, image_path in enumerate(image_paths):
        source_asset_id = str(image_path)
        target_id = f"asset:{index}"
        targets.append(build_asset_target(source_asset_id=source_asset_id, target_id=target_id))
    return targets


def video_path_to_asset_target(video_path: str) -> InferenceTarget:
    return build_asset_target(source_asset_id=str(video_path), target_id="asset:video")


def legacy_frame_payload_to_frame_target(
    source_asset_id: str,
    frame_index: float,
    parent_target_id: Optional[str] = None,
) -> InferenceTarget:
    return build_frame_target(
        source_asset_id=source_asset_id,
        frame_index=frame_index,
        parent_target_id=parent_target_id,
    )


def targets_to_dicts(targets: Iterable[InferenceTarget]) -> List[Dict[str, Any]]:
    return [target.as_dict() for target in targets]


def _build_target_id(scope: str) -> str:
    return f"{scope}:{uuid.uuid4().hex}"
