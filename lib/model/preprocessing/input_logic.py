

import logging
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.server.api_definitions import VideoPathList
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

                #TODO: add configuration option to let people choose whether to reprocess or not for a more accurate version of the same model
                if previousUsedModel.needs_reprocessed(frame_interval, threshold, ai_version, ai_id, ai_filename) == 2:
                    ai_workNeeded = True
                else:
                    skipped_model_categories.add(category)
    return ai_workNeeded, skipped_model_categories