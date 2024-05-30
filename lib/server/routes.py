import asyncio
import logging
from fastapi import HTTPException
from lib.server.api_definitions import ImagePathList, VideoPathList, ImageResult, VideoResult
from lib.server.server_manager import server_manager, app

logger = logging.getLogger("logger")

@app.post("/process_images/")
async def process_images(request: ImagePathList):
    image_paths = request.paths
    logger.info(f"Processing {len(image_paths)} images at paths")
    pipeline_name = request.pipeline_name or server_manager.default_image_pipeline
    futures = [await server_manager.get_request_future([path], pipeline_name) for path in image_paths]
    results = await asyncio.gather(*futures, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            results[i] = {"error": str(result)}

    pipeline = server_manager.pipeline_manager.pipelines[pipeline_name]
    return_result = ImageResult(result=results, pipeline_short_name=pipeline.short_name, pipeline_version=pipeline.version)
    return return_result

@app.post("/process_video/")
async def process_video(request: VideoPathList):
    logger.info(f"Processing video at path: {request.path}")
    pipeline_name = request.pipeline_name or server_manager.default_video_pipeline
    data = [request.path, request.returnTimestamps]
    try:
        future = await server_manager.get_request_future(data, pipeline_name)
        result = await future
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    pipeline = server_manager.pipeline_manager.pipelines[pipeline_name]
    return_result = VideoResult(result=result, pipeline_short_name=pipeline.short_name, pipeline_version=pipeline.version)
    return return_result