from copy import deepcopy
import logging
from lib.model.postprocessing.category_settings import category_config
import lib.model.postprocessing.tag_models as tag_models

logger = logging.getLogger("logger")

def compute_video_tag_info(video_result):
    video_timespans = compute_video_timespans(video_result)
    video_tags, tag_totals = compute_video_tags(video_result)
    return tag_models.VideoTagInfo(video_duration=video_result.metadata.duration, video_tags=video_tags, tag_totals=tag_totals, tag_timespans=video_timespans)

def compute_video_timespans(video_result):
    return compute_video_timespans_OG(video_result)

#TODO Explore other timespan generation methods
def compute_video_timespans_OG(video_result):
    video_duration = video_result.metadata.duration
    toReturn = {}
    for category, tag_raw_timespans in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category settings")
            continue
        frame_interval = float(video_result.metadata.models[category].frame_interval)
        toReturn[category] = {}
        for tag, raw_timespans in tag_raw_timespans.items():
            if tag not in category_config[category]:
                logger.debug(f"Tag {tag} not found in category settings for category {category}")
                continue
            tag_threshold = float(category_config[category][tag]['TagThreshold'])
            tag_max_gap = format_duration_or_percent(category_config[category][tag]['MaxGap'], video_duration)
            tag_min_duration = format_duration_or_percent(category_config[category][tag]['MinMarkerDuration'], video_duration)
            renamed_tag = category_config[category][tag]['RenamedTag']

            tag_timeframes = []
            for raw_timespan in raw_timespans:
                if raw_timespan.confidence < tag_threshold:
                    continue
                
                if not tag_timeframes:
                    tag_timeframes.append(deepcopy(raw_timespan))
                    continue
                else:
                    previous_timeframe = tag_timeframes[-1]
                    if previous_timeframe.end is None:
                        if raw_timespan.start - previous_timeframe.start - frame_interval <= tag_max_gap:
                            previous_timeframe.end = raw_timespan.end or raw_timespan.start
                        else:
                            tag_timeframes.append(deepcopy(raw_timespan))
                    else:
                        if raw_timespan.start - previous_timeframe.end - frame_interval <= tag_max_gap:
                            previous_timeframe.end = raw_timespan.end or raw_timespan.start
                        else:
                            tag_timeframes.append(deepcopy(raw_timespan))
            
            tag_timeframes = [tag_models.TimeFrame(start=timeframe.start, end=timeframe.end) for timeframe in tag_timeframes if timeframe.end is not None and 
                              timeframe.end - timeframe.start >= tag_min_duration]
            if tag_timeframes:
                toReturn[category][renamed_tag] = tag_timeframes
    return toReturn

def compute_video_tags(video_result):
    return compute_video_tags_OG(video_result)

def compute_video_tags_OG(video_result):
    video_tags = {}
    tag_totals = {}
    video_duration = video_result.metadata.duration
    for category, tag_raw_timespans in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category settings")
            continue
        video_tags[category] = set()
        tag_totals[category] = {}
        frame_interval = float(video_result.metadata.models[category].frame_interval)

        for tag, raw_timespans in tag_raw_timespans.items():
            if tag not in category_config[category]:
                logger.debug(f"Tag {tag} not found in category settings for category {category}")
                continue
            required_duration = format_duration_or_percent(category_config[category][tag]['RequiredDuration'], video_duration)
            tag_threshold = float(category_config[category][tag]['TagThreshold'])
            totalDuration = 0.0
            for raw_timespan in raw_timespans:
                if raw_timespan.confidence < tag_threshold:
                    continue
                if raw_timespan.end is None:
                    totalDuration += frame_interval
                else:
                    totalDuration += raw_timespan.end - raw_timespan.start + frame_interval
            tag_totals[category][tag] = totalDuration
            if totalDuration >= required_duration:
                video_tags[category].add(category_config[category][tag]['RenamedTag'])
    return video_tags, tag_totals



def format_duration_or_percent(value, video_duration):
    if isinstance(value, float):
        return value
    elif isinstance(value, str):
        if value.endswith('%'):
            return float(value[:-1]) / 100 * video_duration
        elif value.endswith('s'):
            return float(value[:-1])
        else:
            return float(value)
    elif isinstance(value, int):
        return float(value)