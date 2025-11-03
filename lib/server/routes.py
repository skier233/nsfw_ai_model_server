import asyncio
import json
import logging
from fastapi import HTTPException
from lib.model.postprocessing import tag_models, timeframe_processing
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.model.preprocessing.input_logic import process_video_preprocess
from lib.server.api_definitions import ImagePathList, ImageRequestV3, OptimizeMarkerSettings, VideoPathList, ImageResult, VideoRequestV3, VideoResult
from lib.server.server_manager import server_manager, app, outstanding_requests_middleware
import torch
import time
from lib.model.postprocessing.category_settings import category_config

logger = logging.getLogger("logger")

@app.post("/process_images/")
async def process_images(request: ImagePathList):
    try:
        image_paths = request.paths
        logger.info(f"Processing {len(image_paths)} images")
        pipeline_name = request.pipeline_name or server_manager.default_image_pipeline
        futures = [await server_manager.get_request_future([path, request.threshold, request.return_confidence, None], pipeline_name) for path in image_paths]
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
        logger.debug(f"Returning Video Result for: '{request.path}' Results: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/v3/process_video/")
async def process_video_v3(request: VideoRequestV3):
    try:
        logger.info(f"Processing video in v3 at path: {request.path}")

        pipeline_name = "video_pipeline_dynamic_v3"
        
        data = [request.path, True, request.frame_interval, request.threshold, False, request.vr_video, request.categories_to_skip]

        result = None
        try:
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await future
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/v3/process_images/")
async def process_images_v3(request: ImageRequestV3):
    try:
        image_paths = request.paths
        logger.info(f"Processing {len(image_paths)} images")
        pipeline_name = "image_pipeline_dynamic_v3"
        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        futures = [await server_manager.get_request_future([path, request.threshold, request.return_confidence, None], pipeline_name) for path in image_paths]
        results = await asyncio.gather(*futures, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"error": str(result)}

        models = pipeline.get_ai_models_info()
        return_result = ImageResult(result=results, models=models)
        logger.debug(f"Returning Image Result v3: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing images v3: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v3/current_ai_models/")
async def get_current_video_ai_models():
    try:
        pipeline_name = "video_pipeline_dynamic_v3"
        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        ai_models = pipeline.get_ai_models_info()
        return ai_models
    except Exception as e:
        logger.error(f"Error getting current video AI models: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/optimize_timeframe_settings/")
async def optimize_timeframe_settings(request: OptimizeMarkerSettings):
    try:
        video_result, _ = AIVideoResult.from_client_json(json=request.existing_json_data)

        if video_result is None:
            raise HTTPException(status_code=400, detail="Video Result is None")
        else:
            desired_timespan_data = request.desired_timespan_data
            desired_timespan_category_dict = {}
            renamedtag_category_dict = {}
            for category, category_dict in category_config.items():
                for tag, renamed_tag in category_dict.items():
                    renamedtag_category_dict[renamed_tag["RenamedTag"]] = category
            
            for tag, time_frames in desired_timespan_data.items():
                category = renamedtag_category_dict.get(tag, "Unknown")
                if category not in desired_timespan_category_dict:
                    desired_timespan_category_dict[category] = {}
                time_frames_new = [tag_models.TimeFrame(**(json.loads(time_frame)), totalConfidence=None) for time_frame in time_frames]
                desired_timespan_category_dict[category][tag] = time_frames_new
            timeframe_processing.determine_optimal_timespan_settings(video_result, desired_timespan_data=desired_timespan_category_dict)
        return 
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Return a JSON object with server health information."""
    try:
        info = {}
        info["status"] = "ok"
        info["time"] = time.time()

        # Pipelines
        pipelines = server_manager.pipeline_manager.pipelines
        pipeline_info = {}
        for name, pipeline in pipelines.items():
            try:
                ai_models = pipeline.get_ai_models_info()
                pipeline_info[name] = {
                    "version": getattr(pipeline, "version", None),
                    "short_name": getattr(pipeline, "short_name", None),
                    "ai_models_count": len(ai_models),
                    "ai_models": ai_models,
                }
            except Exception:
                pipeline_info[name] = {"error": "failed to inspect pipeline"}

        info["pipelines"] = pipeline_info

        # Models summary
        total_ai_models = 0
        for p in pipeline_info.values():
            if isinstance(p, dict) and "ai_models_count" in p:
                total_ai_models += p["ai_models_count"]
        info["total_ai_models"] = total_ai_models

        # Outstanding requests
        try:
            info["outstanding_requests"] = outstanding_requests_middleware.outstanding_requests
        except Exception:
            info["outstanding_requests"] = None

        # GPU / CUDA info
        try:
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                try:
                    info["cuda_device_count"] = torch.cuda.device_count()
                except Exception as e:
                    info["cuda_device_count_error"] = str(e)
                try:
                    curr = torch.cuda.current_device()
                    info["current_device"] = curr
                    info["current_device_name"] = torch.cuda.get_device_name(curr)
                except Exception as e:
                    info["cuda_device_error"] = str(e)
                try:
                    info["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated()
                    info["cuda_memory_reserved_bytes"] = torch.cuda.memory_reserved()
                except Exception:
                    pass
        except Exception as e:
            info["cuda_check_error"] = str(e)

        return info
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ready")
async def ready_check():
    """Readiness: returns 200 if pipelines are loaded and server is ready to accept requests."""
    try:
        pipelines = server_manager.pipeline_manager.pipelines
        if pipelines and len(pipelines) > 0:
            return {"ready": True, "loaded_pipelines": list(pipelines.keys())}
        else:
            raise HTTPException(status_code=503, detail="No pipelines loaded")
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(status_code=503, detail=str(e))