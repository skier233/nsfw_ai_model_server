import asyncio
import logging
from typing import Optional
from fastapi import HTTPException
from lib.model.postprocessing import timeframe_processing
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.model.preprocessing.input_logic import process_video_preprocess
from lib.server.api_definitions import ImagePathList, VideoPathList, ImageResult, VideoResult
from lib.server.server_manager import server_manager, app

logger = logging.getLogger("logger")

@app.post("/process_images/")
async def process_images(request: ImagePathList):
    try:
        image_paths = request.paths
        logger.info(f"Processing {len(image_paths)} images")
        pipeline_name = request.pipeline_name or server_manager.default_image_pipeline
        futures = [await server_manager.get_request_future([path, request.threshold, request.return_confidence], pipeline_name) for path in image_paths]
        results = await asyncio.gather(*futures, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"error": str(result)}

        return_result = ImageResult(result=results)
        logger.debug(f"Returning Image Result: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process_video/")
async def process_video(request: VideoPathList):
    try:
        logger.info(f"Processing video at path: {request.path}")
        pipeline_name = request.pipeline_name or server_manager.default_video_pipeline
        

        video_result, json_save_needed = AIVideoResult.from_client_json(json=request.existing_json_data)

        data = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video, None, None]
        if video_result is not None:
            pipeline_to_use = server_manager.pipeline_manager.get_pipeline(pipeline_name)

            #TODO: need to cover the case of a threshold/frame_interval not passed into the request
            ai_work_needed, skipped_categories = process_video_preprocess(video_result, request.frame_interval, request.threshold, pipeline_to_use)
            
            if not ai_work_needed:
                # No models need to run but we may need to update client json and we need to regenerate timespans and tags
                json_result = None
                if json_save_needed:
                    json_result = video_result.to_json()
                return_result = {"json_result": json_result, "video_tag_info": timeframe_processing.compute_video_tag_info(video_result)}

                return VideoResult(result=return_result)
            else:
                # We need to run models, skip ones that aren't needed, and then add to the video_result instead of overwriting it
                data = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video, video_result, skipped_categories]

        try:
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await future
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return_result = VideoResult(result=result)
        logger.debug(f"Returning Video Result: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))