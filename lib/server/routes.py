import asyncio
import logging
from typing import Optional
from fastapi import HTTPException
from lib.server.api_definitions import ImagePathList, ImagePipelineInfo, VideoPathList, ImageResult, VideoPipelineInfo, VideoResult
from lib.server.server_manager import server_manager, app

logger = logging.getLogger("logger")

def get_video_pipeline_config(pipeline_name, frame_interval, threshold, return_confidence):
    pipeline = server_manager.pipeline_manager.pipelines.get(pipeline_name)
    if not pipeline:
        raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} not found")
    if not frame_interval:
        videopreprocessor = pipeline.get_first_video_preprocessor()
        if not videopreprocessor:
            raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} does not have a video preprocessor")
        frame_interval = videopreprocessor.frame_interval
    ai_model = None
    if not threshold:
        ai_model = pipeline.get_first_ai_model()
        if not ai_model:
            raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} does not have an AI model")
        threshold = ai_model.model_threshold
    if return_confidence is None:
        if not ai_model:
            ai_model = pipeline.get_first_ai_model()
        if not ai_model:
            raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} does not have an AI model")
        return_confidence = ai_model.model_return_confidence
    return pipeline.short_name, pipeline.version, frame_interval, threshold, return_confidence

def get_image_pipeline_config(pipeline_name, threshold, return_confidence):
    pipeline = server_manager.pipeline_manager.pipelines.get(pipeline_name)
    if not pipeline:
        raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} not found")
    ai_model = None
    if not threshold:
        ai_model = pipeline.get_first_ai_model()
        if not ai_model:
            raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} does not have an AI model")
        threshold = ai_model.model_threshold
    if return_confidence is None:
        if not ai_model:
            ai_model = pipeline.get_first_ai_model()
        if not ai_model:
            raise HTTPException(status_code=400, detail=f"Pipeline {pipeline_name} does not have an AI model")
        return_confidence = ai_model.model_return_confidence
    return pipeline.short_name, pipeline.version, threshold, return_confidence

@app.get("/video_pipeline_info/")
async def get_video_pipeline_info(
    pipeline_name: Optional[str] = None,
    frame_interval: Optional[float] = None,
    threshold: Optional[float] = None,
    return_confidence: Optional[bool] = None
):
    try:
        pipeline_name = pipeline_name or server_manager.default_video_pipeline
        short_name, version, frame_interval, threshold, return_confidence = get_video_pipeline_config(pipeline_name, frame_interval, threshold, return_confidence)
        return VideoPipelineInfo(pipeline_short_name=short_name, pipeline_version=version, threshold=threshold, frame_interval=frame_interval, return_confidence=return_confidence)
    except Exception as e:
        logger.error(f"Error getting video pipeline info: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/image_pipeline_info/")
async def get_image_pipeline_info(pipeline_name: Optional[str] = None, threshold: Optional[float] = None, return_confidence: Optional[bool] = None):
    try:
        pipeline_name = pipeline_name or server_manager.default_video_pipeline
        short_name, version, threshold, return_confidence = get_image_pipeline_config(pipeline_name, threshold, return_confidence)
        return ImagePipelineInfo(pipeline_short_name=short_name, pipeline_version=version, threshold=threshold, return_confidence=return_confidence)
    except Exception as e:
        logger.error(f"Error getting image pipeline info: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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

        short_name, version, threshold, return_confidence = get_image_pipeline_config(pipeline_name, request.threshold, request.return_confidence)
        return_result = ImageResult(result=results, pipeline_short_name=short_name, pipeline_version=version, threshold=threshold, return_confidence=return_confidence)
        logger.debug(f"Returning Image Result: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process_video/")
async def process_video(request: VideoPathList):
    try:
        logger.info(f"Processing video at path: {request.path}")
        pipeline_name = request.pipeline_name or server_manager.default_video_pipeline
        data = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video]
        try:
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await future
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        short_name, version, frame_interval, threshold, return_confidence = get_video_pipeline_config(pipeline_name, request.frame_interval, request.threshold, request.return_confidence)
        return_result = VideoResult(result=result, pipeline_short_name=short_name, pipeline_version=version, threshold=threshold, frame_interval=frame_interval, return_confidence=return_confidence)
        logger.debug(f"Returning Video Result: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=400, detail=str(e))