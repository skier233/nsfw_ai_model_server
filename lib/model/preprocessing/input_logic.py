import logging
from lib.model.postprocessing.post_processing_settings import post_processing_config
logger = logging.getLogger("logger")

def process_video_preprocess(video_result, frame_interval, threshold, pipeline):
    ai_workNeeded = False
    ai_models_info = pipeline.get_ai_models_info()

    skipped_model_categories = set()
    previouslyUsedModelsDict = video_result.metadata.models
    for ai_version, ai_id, ai_filename, ai_categories in ai_models_info:
        for category in ai_categories:
            if category not in previouslyUsedModelsDict:
                ai_workNeeded = True
            else:
                previousUsedModel = previouslyUsedModelsDict[category]

                needs_reprocessed = previousUsedModel.needs_reprocessed(frame_interval, threshold, ai_version, ai_id, ai_filename)
                if post_processing_config.get('reprocess_with_more_accurate_same_model', False) and needs_reprocessed == 1:
                    ai_workNeeded = True
                elif  needs_reprocessed == 2:
                    ai_workNeeded = True
                else:
                    skipped_model_categories.add(category)
    return ai_workNeeded, skipped_model_categories