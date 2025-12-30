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
            # Store excluded_tags in the future data if provided
            excluded_tags = request.excluded_tags if request.excluded_tags else []
            # Always set it, even if empty, to ensure consistent behavior
            await future.set_data("excluded_tags", excluded_tags)
            result = await future
            logger.debug(f"Video v3 processing result: {result}")
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
        excluded_tags = request.excluded_tags if request.excluded_tags else []
        futures = []
        for path in image_paths:
            future = await server_manager.get_request_future([path, request.threshold, request.return_confidence, None], pipeline_name)
            # Store excluded_tags in the future data so postprocessor can access it
            # Always set it, even if empty, to ensure consistent behavior
            await future.set_data("excluded_tags", excluded_tags)
            futures.append(future)
        results = await asyncio.gather(*futures, return_exceptions=True)

        aggregate_metrics = {
            "preprocess_seconds": 0.0,
            "ai_inference_seconds": 0.0,
            "image_count": 0,
        }
        preprocess_backends = set()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"error": str(result)}
                continue

            if isinstance(result, dict) and "metrics" in result:
                metrics = result.get("metrics") or {}
                aggregate_metrics["preprocess_seconds"] += metrics.get("preprocess_seconds", 0.0)
                aggregate_metrics["ai_inference_seconds"] += metrics.get("ai_inference_seconds", 0.0)
                backend = metrics.get("preprocess_backend")
                if backend:
                    preprocess_backends.add(backend)
                aggregate_metrics["image_count"] += 1
                inner_result = result.get("result")
                results[i] = inner_result if inner_result is not None else result
            else:
                results[i] = result

        models = pipeline.get_ai_models_info()
        aggregate_metrics["ai_model_count"] = len(models)
        aggregate_metrics["total_runtime_seconds"] = (
            aggregate_metrics["preprocess_seconds"] + aggregate_metrics["ai_inference_seconds"]
        )
        if aggregate_metrics["total_runtime_seconds"] > 0 and aggregate_metrics["image_count"] > 0:
            aggregate_metrics["images_per_second"] = aggregate_metrics["image_count"] / aggregate_metrics["total_runtime_seconds"]
        if preprocess_backends:
            aggregate_metrics["preprocess_backend"] = preprocess_backends.pop() if len(preprocess_backends) == 1 else "mixed"

        return_payload = {
            "result": results,
            "models": models,
            "metrics": aggregate_metrics,
        }
        logger.debug(f"Returning Image Result v3: {return_payload}")
        return return_payload
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

@app.get("/tags/available")
async def get_available_tags():
    """Returns tags grouped by model from all active AI models."""
    try:
        from lib.model.ai_model import AIModel
        models_dict = {}  # Use dict to deduplicate by model name
        all_tags_set = set()
        
        pipelines = server_manager.pipeline_manager.pipelines
        if not pipelines:
            return {"tags": [], "models": []}
        
        for pipeline_name, pipeline in pipelines.items():
            for model_wrapper in pipeline.models:
                if isinstance(model_wrapper.model.model, AIModel):
                    ai_model = model_wrapper.model.model
                    # Check if tags are loaded (they should be after model.load() is called)
                    if hasattr(ai_model, 'tags') and ai_model.tags:
                        # tags is a dict mapping index -> tag_name
                        model_tags = [tag_name for tag_name in ai_model.tags.values()]
                        model_tags_sorted = sorted(model_tags)
                        
                        # Add to all tags set
                        all_tags_set.update(model_tags)
                        
                        # Get model info
                        model_name = ai_model.model_file_name
                        
                        # Deduplicate: if we've seen this model before, merge tags (in case same model in multiple pipelines)
                        if model_name in models_dict:
                            # Merge tags and keep unique
                            existing_tags = set(models_dict[model_name]["tags"])
                            new_tags = set(model_tags_sorted)
                            merged_tags = sorted(list(existing_tags | new_tags))
                            models_dict[model_name]["tags"] = merged_tags
                            models_dict[model_name]["tagCount"] = len(merged_tags)
                        else:
                            # First time seeing this model
                            model_categories = ai_model.model_category or []
                            model_identifier = ai_model.model_identifier
                            model_version = ai_model.model_version
                            
                            models_dict[model_name] = {
                                "name": model_name,
                                "identifier": model_identifier,
                                "version": model_version,
                                "categories": model_categories,
                                "tags": model_tags_sorted,
                                "tagCount": len(model_tags_sorted)
                            }
        
        # Convert dict to list and mark all as active (since they're loaded)
        models_data = list(models_dict.values())
        for model in models_data:
            model["active"] = True
        
        # Get list of all available models (including inactive ones) for comparison
        from lib.configurator.configure_active_ai import load_available_ai_models, load_active_ai_models
        all_available_models = load_available_ai_models()
        active_model_names = set(load_active_ai_models())
        
        # Add inactive models to the response (without tags, but with metadata)
        active_model_names_in_response = {m["name"] for m in models_data}
        for available_model in all_available_models:
            model_name = available_model.get('yaml_file_name', '')
            if model_name and model_name not in active_model_names_in_response:
                # This model exists but is not active
                model_categories = available_model.get('model_category', [])
                # Generate a display name from the model name
                display_name = model_name.replace('_', ' ').title()
                models_data.append({
                    "name": model_name,
                    "displayName": display_name,  # Add display name for frontend
                    "identifier": available_model.get('model_identifier'),
                    "version": available_model.get('model_version'),
                    "categories": model_categories,
                    "tags": [],  # No tags since model is not loaded
                    "tagCount": 0,
                    "active": False
                })
        
        # Return both: all unique tags (for backward compatibility) and tags by model
        all_tags_list = sorted(list(all_tags_set))
        logger.debug(f"Returning {len(all_tags_list)} unique tags across {len([m for m in models_data if m.get('active')])} active models, {len(models_data)} total models")
        result = {
            "tags": all_tags_list,  # Backward compatibility
            "models": models_data    # New: tags grouped by model, includes active/inactive status
        }
        return result
    except Exception as e:
        logger.error(f"Error getting available tags: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))