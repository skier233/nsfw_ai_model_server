from typing import Any, List
from pydantic import BaseModel

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

class VideoResult(BaseModel):
    result: Any
    pipeline_short_name: str
    pipeline_version: float
    threshold: float
    frame_interval: int
    return_confidence: bool

class ImageResult(BaseModel):
    result: Any
    pipeline_short_name: str
    pipeline_version: float
    threshold: float
    return_confidence: bool