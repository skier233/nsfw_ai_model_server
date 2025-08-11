import gzip
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("logger")

class TagTimeFrame(BaseModel):
    start: float
    end: Optional[float] = None
    confidence: Optional[float] = None
    def __str__(self):
        return f"TagTimeFrame(start={self.start}, end={self.end}, confidence={self.confidence})"

class ModelInfo(BaseModel):
    frame_interval: float
    threshold: float
    version: float
    ai_model_id: int
    file_name: Optional[str]
    
    def needs_reprocessed(self, new_frame_interval, new_threshold, new_version, new_ai_model_id, new_file_name):
        # 0 = new model/config is the same or worse than current, 1 = new model/config is better version of same version, 2 = new model or different config
        model_toReturn = -1

        if new_file_name == self.file_name and new_version == self.version:
            model_toReturn = 0
        elif new_version == self.version and new_ai_model_id < self.ai_model_id and self.ai_model_id >= 950:
            model_toReturn = 2
        elif new_version == self.version and new_ai_model_id < self.ai_model_id:
            model_toReturn = 1
        elif new_version == self.version and new_ai_model_id >= self.ai_model_id:
            model_toReturn = 0
        else:
            model_toReturn = 2

        same_config = True

        if new_frame_interval % self.frame_interval != 0 or new_threshold < self.threshold:
            same_config = False

        if same_config:
            return model_toReturn
        else:
            return 2



    def __str__(self):
        return f"ModelInfo(frame_interval={self.frame_interval}, threshold={self.threshold}, version={self.version}, ai_model_id={self.model_id}, file_name={self.file_name})"

class VideoMetadata(BaseModel):
    duration: float
    models: Dict[str, ModelInfo]
    def __str__(self):
        return f"VideoMetadata(duration={self.duration}, models={self.models})"


class AIVideoResult(BaseModel):
    schema_version: int
    metadata: VideoMetadata
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]]

    def to_json(self):
        return self.model_dump_json(exclude_none=True)

    def add_server_result(self, server_result):
        ai_version_and_ids = server_result['ai_models_info']
        updated_categories = set()
        current_models = self.metadata.models

        frame_interval = server_result['frame_interval']
        threshold = server_result['threshold']
        for ai_version, ai_id, ai_filename, ai_categories in ai_version_and_ids:
            for category in ai_categories:
                if category in current_models:
                    model_info = current_models[category]
                    if model_info.needs_reprocessed(frame_interval, threshold, ai_version, ai_id, ai_filename) > 0:
                        current_models[category] = ModelInfo(frame_interval=frame_interval, threshold=threshold, version=ai_version, ai_model_id=ai_id, file_name=ai_filename)
                        updated_categories.add(category)
                else:
                    current_models[category] = ModelInfo(frame_interval=frame_interval, threshold=threshold, version=ai_version, ai_model_id=ai_id, file_name=ai_filename)
                    updated_categories.add(category)

        frames = server_result['frames']
        timespans = AIVideoResult.__mutate_server_result_tags(frames, frame_interval)
        logger.debug(f"Updated categories: {updated_categories}")
        for category in updated_categories:
            self.timespans[category] = timespans[category]
    
    @classmethod
    def from_client_json(cls, json):
        if json is None:
            return None, True
        if "schema_version" not in json:
            from lib.model.postprocessing.AI_VideoResultV0 import AIVideoResultV0
            v0 = AIVideoResultV0(**json)
            return v0.to_V1(), True
        else:
            return cls(**json), False

    @classmethod
    def from_server_result(cls, server_result):
        frames = server_result['frames']
        video_duration = server_result['video_duration']
        frame_interval = server_result['frame_interval']
        timespans = AIVideoResult.__mutate_server_result_tags(frames, frame_interval)
        ai_version_and_ids = server_result['ai_models_info']
        modelinfos = {}
        for ai_version, ai_id, ai_filename, ai_categories in ai_version_and_ids:
            model_info = ModelInfo(frame_interval=frame_interval, threshold=server_result['threshold'], version=ai_version, ai_model_id=ai_id, file_name=ai_filename)
            for category in ai_categories:
                if category in modelinfos:
                    raise Exception(f"Category {category} already exists in modelinfos. Models should not have overlapping categories!")
                modelinfos[category] = model_info
        metadata = VideoMetadata(duration=video_duration, models=modelinfos)
        schema_version = 1
        return cls(schema_version=schema_version, metadata=metadata, timespans=timespans)

    @classmethod
    def __mutate_server_result_tags(cls, frames, frame_interval):
        toReturn = {}
        for frame in frames:
            frame_index = frame['frame_index']
            for key, value in frame.items():
                if key != "frame_index":
                    currentCategoryDict = None
                    if not isinstance(value, list):
                        raise Exception(f"Category {key} is not a list")
                    if key in toReturn:
                        currentCategoryDict = toReturn[key]
                    else:
                        currentCategoryDict = {}
                        toReturn[key] = currentCategoryDict
                    
                    for item in value:
                        if isinstance(item, tuple):
                            tag_name, confidence = item
                        else:
                            tag_name = item
                            confidence = None

                        if tag_name not in currentCategoryDict:
                            currentCategoryDict[tag_name] = [TagTimeFrame(start=frame_index, end=None, confidence=confidence)]
                        else:
                            last_time_frame = currentCategoryDict[tag_name][-1]

                            if last_time_frame.end is None:
                                if frame_index - last_time_frame.start == frame_interval and last_time_frame.confidence == confidence:
                                    last_time_frame.end = frame_index
                                else:
                                    currentCategoryDict[tag_name].append(TagTimeFrame(start=frame_index, end=None, confidence=confidence))
                            elif last_time_frame.confidence == confidence and frame_index - last_time_frame.end == frame_interval:
                                last_time_frame.end = frame_index
                            else:
                                currentCategoryDict[tag_name].append(TagTimeFrame(start=frame_index, end=None, confidence=confidence))

        return toReturn