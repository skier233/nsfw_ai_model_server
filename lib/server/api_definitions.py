from typing import Any, List
from pydantic import BaseModel

class ImagePathList(BaseModel):
    paths: List[str]
    pipeline_name: str = None

class VideoPathList(BaseModel):
    path: str
    returnTimestamps: bool = True
    pipeline_name: str = None

class VideoResult(BaseModel):
    result: Any
    pipeline_short_name: str
    pipeline_version: float

class ImageResult(BaseModel):
    result: Any
    pipeline_short_name: str
    pipeline_version: float