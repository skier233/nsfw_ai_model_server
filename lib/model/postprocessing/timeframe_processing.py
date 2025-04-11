from copy import deepcopy
import logging
import math
from lib.model.postprocessing.category_settings import category_config
import lib.model.postprocessing.tag_models as tag_models
from lib.model.postprocessing.post_processing_settings import get_or_default, post_processing_config

logger = logging.getLogger("logger")

def compute_video_tag_info(video_result):
    video_timespans = compute_video_timespans(video_result)
    video_tags, tag_totals = compute_video_tags(video_result)
    return tag_models.VideoTagInfo(video_duration=video_result.metadata.duration, video_tags=video_tags, tag_totals=tag_totals, tag_timespans=video_timespans)

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
                #logger.debug(f"Tag {tag} not found in category settings for category {category}")
                continue
            
            tag_min_duration = format_duration_or_percent(get_or_default(category_config[category][tag], 'MinMarkerDuration', 12), video_duration)
            if tag_min_duration <= 0:
                #logger.debug(f"Tag {tag} has a min duration of less than 0, skipping")
                continue
            tag_threshold = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            tag_max_gap = format_duration_or_percent(get_or_default(category_config[category][tag], 'MaxGap', 6), video_duration)
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
            
            tag_timeframes = [tag_models.TimeFrame(start=timeframe.start, end=timeframe.end, totalConfidence=None) for timeframe in tag_timeframes if timeframe.end is not None and 
                              timeframe.end - timeframe.start >= tag_min_duration]
            if tag_timeframes:
                toReturn[category][renamed_tag] = tag_timeframes
    return toReturn

def compute_video_timespans_clustering(video_result, density_weight, gap_factor, average_factor, min_gap):
    video_duration = video_result.metadata.duration
    toReturn = {}

    for category, tag_raw_timespans in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category config")
            continue
        
        frame_interval = float(video_result.metadata.models[category].frame_interval)
        toReturn[category] = {}
        
        for tag, raw_timespans in tag_raw_timespans.items():
            if tag not in category_config[category]:
                #logger.debug(f"Tag {tag} not found in category config for {category}")
                continue
            
            tag_threshold = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            renamed_tag = category_config[category][tag]['RenamedTag']
            tag_min_duration = format_duration_or_percent(
                get_or_default(category_config[category][tag], 'MinMarkerDuration', 12),
                video_duration
            )
            if tag_min_duration <= 0:
                #logger.debug(f"Tag {tag} has a min duration of less than 0, skipping")
                continue
            
            # Get initial buckets by merging adjacent timeframes that are immediately next to each other and keeping others alone
            initial_buckets = []
            current_bucket = None
            for i, raw_timespan in enumerate(raw_timespans):
                confidence = raw_timespan.confidence
                if confidence < tag_threshold:
                    continue

                start = raw_timespan.start
                end = raw_timespan.end or raw_timespan.start
                
                duration = (end - start) + frame_interval
                if current_bucket is None:
                    current_bucket = tag_models.TimeFrame(start=start, end=end, totalConfidence=confidence *  duration)
                else:
                    if start - current_bucket.end == frame_interval:
                        current_bucket.merge(start, end, confidence, frame_interval)
                    else:
                        initial_buckets.append(current_bucket)
                        current_bucket = tag_models.TimeFrame(start=start, end=end, totalConfidence=confidence * duration)
            # Append the last bucket if it exists
            if current_bucket is not None:
                initial_buckets.append(current_bucket)
            
            # Iterative merging process
            def should_merge(current_bucket, next_bucket, frame_interval, density_weight = density_weight, gap_factor = gap_factor, average_factor = average_factor, min_gap=min_gap):
                gap = next_bucket.start - current_bucket.end - frame_interval
                duration_current = current_bucket.get_duration(frame_interval)
                duration_next = next_bucket.get_duration(frame_interval)
                density_current = current_bucket.get_density(frame_interval)
                density_next = next_bucket.get_density(frame_interval)

                weighted_duration_current = duration_current * (1 + density_weight * density_current)
                weighted_duration_next = duration_next * (1 + density_weight * density_next)

                weighted_diff = abs(weighted_duration_current - weighted_duration_next)
                # Merge condition based on density and gap
                merge_condition = (gap <= min_gap + (min(weighted_duration_current, weighted_duration_next) + weighted_diff * average_factor) * gap_factor)
                return merge_condition

            # Perform iterative merging
            merged_buckets = initial_buckets
            merging_occurred = True

            max_iterations = 10
            iterations = 0
            while merging_occurred and iterations < max_iterations:
                merging_occurred = False
                new_buckets = []
                i = 0
                iterations += 1

                while i < len(merged_buckets):
                    if i < len(merged_buckets) - 1 and should_merge(merged_buckets[i], merged_buckets[i + 1], frame_interval):
                        # Merge the current and next bucket
                        current_bucket = merged_buckets[i]
                        next_bucket = merged_buckets[i + 1]
                        new_bucket = tag_models.TimeFrame(
                            start=current_bucket.start,
                            end=next_bucket.end,
                            totalConfidence=current_bucket.totalConfidence + next_bucket.totalConfidence
                        )
                        new_buckets.append(new_bucket)
                        i += 2  # Skip next as it's merged
                        merging_occurred = True
                    else:
                        new_buckets.append(merged_buckets[i])
                        i += 1
                merged_buckets = new_buckets

            # Filter out clusters that do not meet the minimum duration requirement
            final_buckets = [
                bucket for bucket in merged_buckets 
                if (bucket.get_duration(frame_interval)) >= tag_min_duration
            ]

            # Store results in the toReturn dictionary
            toReturn[category][renamed_tag] = final_buckets

    return toReturn

def compute_video_timespans_proportional_merge(video_result, prop=0.5):
    """
    prop is the fraction of the last timespan's duration which we allow as an interruption 
    before merging instead of creating a new timespan.
    """
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
                #logger.debug(f"Tag {tag} not found in category settings for category {category}")
                continue
            
            tag_threshold = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            tag_max_gap = format_duration_or_percent(get_or_default(category_config[category][tag], 'MaxGap', 6), video_duration)
            tag_min_duration = format_duration_or_percent(get_or_default(category_config[category][tag], 'MinMarkerDuration', 12), video_duration)
            renamed_tag = category_config[category][tag]['RenamedTag']

            if tag_min_duration <= 0:
                #logger.debug(f"Tag {tag} has a min duration of less than 0, skipping")
                continue

            tag_timeframes = []
            for raw_timespan in raw_timespans:
                if raw_timespan.confidence < tag_threshold:
                    continue
                
                # If no timespans yet, start a new one
                if not tag_timeframes:
                    tag_timeframes.append(deepcopy(raw_timespan))
                    continue
                
                previous_timeframe = tag_timeframes[-1]
                if previous_timeframe.end is None:
                    # Original gap check
                    if raw_timespan.start - previous_timeframe.start - frame_interval <= tag_max_gap:
                        previous_timeframe.end = raw_timespan.end or raw_timespan.start
                    else:
                        gap = (raw_timespan.start - previous_timeframe.start) - frame_interval
                        last_duration = (previous_timeframe.end or previous_timeframe.start) - previous_timeframe.start
                        # Check the proportional gap
                        if gap <= prop * last_duration:
                            previous_timeframe.end = raw_timespan.end or raw_timespan.start
                        else:
                            tag_timeframes.append(deepcopy(raw_timespan))
                else:
                    # If the previous timespan has an end, compare the gap to 'previous_timeframe.end'
                    gap = (raw_timespan.start - previous_timeframe.end) - frame_interval
                    if gap <= tag_max_gap:
                        previous_timeframe.end = raw_timespan.end or raw_timespan.start
                    else:
                        last_duration = previous_timeframe.end - previous_timeframe.start
                        if gap <= prop * last_duration:
                            previous_timeframe.end = raw_timespan.end or raw_timespan.start
                        else:
                            tag_timeframes.append(deepcopy(raw_timespan))

            # Filter timespans by min duration
            tag_timeframes = [
                tag_models.TimeFrame(start=tf.start, end=tf.end)
                for tf in tag_timeframes 
                if tf.end is not None and (tf.end - tf.start) >= tag_min_duration
            ]
            if tag_timeframes:
                toReturn[category][renamed_tag] = tag_timeframes
    return toReturn


active_timespan_method = "Clustering"
timespan_methods = {
    "OG": compute_video_timespans_OG,
    "Clustering": compute_video_timespans_clustering,
    "Proportional_Merge": compute_video_timespans_proportional_merge
}

timespan_configuration_defaults = {
    "OG": [0],
    "Clustering": [0.04, 0.85, 0.25, 0.0],
    "Proportional_Merge": []
}

timespan_configuration_sweep = {
    "OG": [],
    "Clustering": [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2], [0.7, 0.8, 0.93, 0.85, 0.87, 0.9, 0.95, 1.0], [0.00, 0.1, 0.15, 0.17, 0.2, 0.225, 0.25, 0.275, 0.3, 0.4], [0.0, 1.0, 2.0, 4.0, 6.0]],
    "Proportional_Merge": []
}

def load_post_processing_settings(config):
    timespan_systems = config.get("timespan_generation_systems", {})
    if not timespan_systems:
        return
    
    for method, settings in timespan_systems.items():
        if method in timespan_methods:
            timespan_configuration_defaults[method] = settings.get("defaults", timespan_configuration_defaults[method])
            timespan_configuration_sweep[method] = settings.get("sweep", timespan_configuration_sweep[method])

load_post_processing_settings(post_processing_config)

def compute_video_timespans(video_result, method=timespan_methods[active_timespan_method], params=timespan_configuration_defaults[active_timespan_method]):
    try:
        return method(video_result, *params)
        #return compute_video_timespans_clustering(video_result)
    except Exception as e:
        logger.error(f"Error in compute_video_timespans: {e}")
        logger.error("Stack trace:", exc_info=True)
        return {}
import itertools

def determine_optimal_timespan_settings(video_result, desired_timespan_data):
    """
    1) Sweeps through timespan_configuration_sweep[active_timespan_method] to generate parameter combos.
    2) Calls compute_video_timespans(...) with each combo.
    3) Measures how close the result is to desired_timespan_data by summing, in seconds:
       - The time where actual has a tag but desired doesn't
         plus
       - The time where desired has a tag but actual doesn't.
       (This is the symmetric difference in time.)
    4) Prints the 10 best (lowest-loss) settings with their losses.
    """

    def total_tag_duration(spans):
        """Sum the duration of merged intervals."""
        return sum((m.end - m.start) for m in spans)

    def intersection_coverage(A, B):
        """
        Compute how many seconds A and B overlap.
        Assumes A and B are sorted & merged.
        """
        i, j = 0, 0
        overlap = 0.0
        while i < len(A) and j < len(B):
            startA, endA = A[i].start, A[i].end
            startB, endB = B[j].start, B[j].end

            inter_start = max(startA, startB)
            inter_end = min(endA, endB)
            if inter_start < inter_end:
                overlap += (inter_end - inter_start)

            # Advance whichever ends first
            if endA < endB:
                i += 1
            else:
                j += 1

        return overlap

    def measure_loss(actual_timespans, desired_timespans):
        """
        For each (category, tag), calculate the total number of seconds where
        actual has a tag but desired doesn't OR desired has a tag but actual doesn't.
        (i.e. the symmetric difference in time.)
        """
        shared_categories = set(actual_timespans.keys()).union(desired_timespans.keys())
        total_mismatch = 0.0

        for cat in shared_categories:
            actual_tag_timeframes = actual_timespans.get(cat, {})
            desired_tags = desired_timespans.get(cat, {})

            all_tags = set(desired_tags.keys())
            for tag in all_tags:
                actual_tag_timeframes_list = actual_tag_timeframes.get(tag, [])
                desired_tag_timeframes_list = desired_tags.get(tag, [])

                actual_total_tag_time = total_tag_duration(actual_tag_timeframes_list)
                desired_total_tag_time = total_tag_duration(desired_tag_timeframes_list)

                if actual_total_tag_time == 0.0 and desired_total_tag_time == 0.0:
                    mismatch = 0.0
                else:
                    inter = intersection_coverage(actual_tag_timeframes_list, desired_tag_timeframes_list)
                    mismatch = (actual_total_tag_time + desired_total_tag_time) - 2.0 * inter  # Symmetric difference
                total_mismatch += mismatch

        return total_mismatch
    
    def mismatch_count(actual_timespans, desired_timespans):
        actual_spans = sum(len(spans) for cat_dict in actual_timespans.values() for spans in cat_dict.values())
        desired_spans = sum(len(spans) for cat_dict in desired_timespans.values() for spans in cat_dict.values())
        return abs(actual_spans - desired_spans)

    # We assume desired_timespan_data follows the same structure as actual output:
    # {category -> {tag -> [TimeFrame, ...]}}

    used_method = timespan_methods[active_timespan_method]
    sweep_lists = timespan_configuration_sweep[active_timespan_method]

    if not sweep_lists:
        logger.debug("No sweep lists defined for the active method.")
        return

    param_combos = list(set(itertools.product(*sweep_lists)))
    results = []

    for combo in param_combos:
        try:
            # Call the method with these params
            timespans_output = used_method(video_result, *combo)
            combo_loss = measure_loss(timespans_output, desired_timespan_data)
            total_spans = sum(len(spans) for cat_dict in timespans_output.values() for spans in cat_dict.values())
            mismatch_total = mismatch_count(timespans_output, desired_timespan_data)
            results.append((combo_loss + 2 * mismatch_total, combo, total_spans, combo_loss, mismatch_total))
        except Exception as e:
            logger.debug(f"Error with combo {combo}: {e}")
            logger.debug("Stack trace:", exc_info=True)

    # Sort by mismatch ascending: lower means closer to desired
    results.sort(key=lambda x: x[0])

    logger.info("Top 10 parameter combos (lowest mismatch):")
    for i, (loss_val, combo, total_spans, seconds_diff, mismatch_total) in enumerate(results[:10]):
        logger.info(f"#{i+1} Loss={loss_val:.4f} | Timespans={total_spans} | Params={combo} | SecondsDiff={seconds_diff:.2f} | MismatchCount={mismatch_total}")

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
            
            required_duration = format_duration_or_percent(get_or_default(category_config[category][tag], 'RequiredDuration', 20), video_duration)
            tag_threshold = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            totalDuration = 0.0
            for raw_timespan in raw_timespans:
                if raw_timespan.confidence and raw_timespan.confidence < tag_threshold:
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
    try:
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
    except Exception as e:
        logger.error(f"Error in format_duration_or_percent: {e}")
        logger.debug("Stack trace:", exc_info=True)
        return 0.0