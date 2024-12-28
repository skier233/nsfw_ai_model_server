
from typing import Dict, List, Set
from pydantic import BaseModel

class TimeFrame(BaseModel):
    start: float
    end: float

    def __str__(self):
        return f"TimeFrame(start={self.start}, end={self.end})"
    
class VideoTagInfo(BaseModel):
    video_duration: float
    video_tags: Dict[str, Set[str]]
    tag_totals: Dict[str, Dict[str, float]]
    tag_timespans: Dict[str, Dict[str, List[TimeFrame]]]

    def to_json(self):
        return self.model_dump_json(exclude_none=True)

    def __str__(self):
        return f"VideoTagInfo(video_duration={self.video_duration}, video_tags={self.video_tags}, tag_totals={self.tag_totals}, tag_timespans={self.tag_timespans})"