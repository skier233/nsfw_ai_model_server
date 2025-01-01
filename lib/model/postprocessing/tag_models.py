
from typing import Dict, List, Optional, Set
from pydantic import BaseModel

class TimeFrame(BaseModel):
    start: float
    end: float
    totalConfidence: Optional[float]

    def get_density(self, frame_interval):
        return self.totalConfidence / (self.get_duration(frame_interval))
    
    def get_duration(self, frame_interval):
        return (self.end - self.start) + frame_interval
    
    def merge(self, new_start, new_end, new_confidence, frame_interval):
        self.start = min(self.start, new_start)
        self.end = max(self.end, new_end)
        self.totalConfidence += new_confidence * (self.get_duration(frame_interval))

    def __str__(self):
        return f"TimeFrame(start={self.start}, end={self.end}, totalConfidence={self.totalConfidence})"
    
class VideoTagInfo(BaseModel):
    video_duration: float
    video_tags: Dict[str, Set[str]]
    tag_totals: Dict[str, Dict[str, float]]
    tag_timespans: Dict[str, Dict[str, List[TimeFrame]]]

    def to_json(self):
        return self.model_dump_json(exclude_none=True)

    def __str__(self):
        return f"VideoTagInfo(video_duration={self.video_duration}, video_tags={self.video_tags}, tag_totals={self.tag_totals}, tag_timespans={self.tag_timespans})"