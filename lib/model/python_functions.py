
import asyncio
import logging
from lib.config.config_utils import load_config
from lib.model.preprocessing_python.image_preprocessing import get_video_duration_decord
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
import lib.model.postprocessing.timeframe_processing as timeframe_processing
from lib.model.postprocessing.category_settings import category_config
from lib.model.skip_input import Skip
from lib.model.postprocessing.post_processing_settings import post_processing_config
logger = logging.getLogger("logger")

async def result_coalescer(data):
    for item in data:
        itemFuture = item.item_future
        result = {}
        for input_name in item.input_names:
            ai_result = itemFuture[input_name]
            if not isinstance(ai_result, Skip):
                result[input_name] = itemFuture[input_name]
        await itemFuture.set_data(item.output_names[0], result)
        
async def result_finisher(data):
    for item in data:
        itemFuture = item.item_future
        future_results = itemFuture[item.input_names[0]]
        itemFuture.close_future(future_results)

async def batch_awaiter(data):
    for item in data:
        itemFuture = item.item_future
        futures = itemFuture[item.input_names[0]]
        results = await asyncio.gather(*futures, return_exceptions=True)
        await itemFuture.set_data(item.output_names[0], results)

async def video_result_postprocessor(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_decord(itemFuture[item.input_names[1]])
        result = {"frames": itemFuture[item.input_names[0]], "video_duration": duration, "frame_interval": float(itemFuture[item.input_names[2]]), "threshold": float(itemFuture[item.input_names[3]]), "ai_models_info": itemFuture['pipeline'].get_ai_models_info()}
        del itemFuture.data["pipeline"]

        videoResult = itemFuture[item.input_names[4]]
        if videoResult is not None:
            videoResult.add_server_result(result)
        else:
            videoResult = AIVideoResult.from_server_result(result)

        toReturn = {"json_result": videoResult.to_json(), "video_tag_info": timeframe_processing.compute_video_tag_info(videoResult)}
        
        await itemFuture.set_data(item.output_names[0], toReturn)

async def image_result_postprocessor(data):
    toReturn = {}
    for item in data:
        itemFuture = item.item_future
        result = itemFuture[item.input_names[0]]
        for category, tags in result.items():
            if category not in category_config:
                continue
            toReturn[category] = []
            for tag in tags:
                if isinstance(tag, tuple):
                    tagname, confidence = tag
                    if tagname not in category_config[category]:
                        continue
                    tag_threshold = float(category_config[category][tagname]['TagThreshold'])
                    renamed_tag = category_config[category][tagname]['RenamedTag']

                    if not post_processing_config["use_category_image_thresholds"]:
                        toReturn[category].append((renamed_tag, confidence))
                    elif confidence >= tag_threshold:
                        toReturn[category].append((renamed_tag, confidence))
                else:
                    if tag not in category_config[category]:
                        continue
                    renamed_tag = category_config[category][tag]['RenamedTag']
                    toReturn[category].append(renamed_tag)


        await itemFuture.set_data(item.output_names[0], toReturn)