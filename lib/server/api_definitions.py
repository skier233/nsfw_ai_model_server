from typing import Any, Dict, List
from pydantic import BaseModel

from lib.model.postprocessing import tag_models

class ImagePathList(BaseModel):
    paths: List[str]
    pipeline_name: str = None
    threshold: float = None
    return_confidence: bool = None

class VideoPathList(BaseModel):
    path: str
    returnTimestamps: bool = True
    pipeline_name: str = None
    frame_interval: float = None
    threshold: float = None
    return_confidence: bool = None
    vr_video: bool = False
    existing_json_data: Any = None

class VideoRequestV3(BaseModel):
    path: str
    categories_to_skip: List[str] = None
    frame_interval: float = None
    threshold: float = None
    vr_video: bool = False
    categories_to_skip: List[str] = None

class OptimizeMarkerSettings(BaseModel):
    existing_json_data: Any = None
    desired_timespan_data: Any = None

class VideoResult(BaseModel):
    result: Any

class ImageResult(BaseModel):
    result: Any

class AIModelInfo(BaseModel):
    name: str
    identifier: int
    version: float
    categories: List[str]
    type: str