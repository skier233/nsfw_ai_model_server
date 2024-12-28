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
    vr_video: bool = False
    existing_json_data: Any = None

class VideoResult(BaseModel):
    result: Any

class ImageResult(BaseModel):
    result: Any