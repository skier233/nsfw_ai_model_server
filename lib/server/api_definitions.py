from typing import List
from pydantic import BaseModel

class ImagePathList(BaseModel):
    paths: List[str]

class VideoPathList(BaseModel):
    path: str
    returnTimestamps: bool = True