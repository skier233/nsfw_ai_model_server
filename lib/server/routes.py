import asyncio
import json
import logging
import os
import numpy as np
from fastapi import HTTPException
from lib.model.postprocessing import tag_models, timeframe_processing
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.model.preprocessing.input_logic import process_video_preprocess
from lib.server.api_definitions import (
    AudioRequest,
    AudioRequestV4,
    CustomPipelineRequestV4,
    ImagePathList,
    ImageRequestV3,
    ImageRequestV4,
    ModelSelectionRequestV4,
    OptimizeMarkerSettings,
    TextEncodeRequestV4,
    VideoPathList,
    ImageResult,
    VideoRequestV3,
    VideoRequestV4,
    VideoResult,
)
from lib.server.text_encoding_service import text_encoding_service
from lib.server.server_manager import server_manager, app, outstanding_requests_middleware
from lib.server.v4_service import (
    VALID_LOAD_POLICIES,
    expand_requested_models_with_dependencies,
    filter_pipeline_models,
    get_capability_catalog,
    get_loaded_models,
    get_model_catalog,
    load_models,
    resolve_request_models,
    unload_models,
)
import torch
import time
from lib.model.postprocessing.category_settings import category_config


def _sanitize_for_log(obj, _depth=0):
    """Return a lightweight copy of *obj* suitable for debug logging.

    Embedding vectors (numpy arrays / long lists of floats) are replaced
    with a short summary so they don't flood the console.  Pydantic
    models are converted to dicts before traversal.
    """
    if _depth > 12:
        return "..."
    # Pydantic models → dict so we can recurse normally.
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(exclude_none=True)
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, torch.Tensor):
        return f"<Tensor shape={list(obj.shape)} dtype={obj.dtype}>"
    if isinstance(obj, (list, tuple)):
        # Heuristic: a long list of numbers is almost certainly an embedding.
        if len(obj) > 32 and all(isinstance(v, (int, float)) for v in obj[:8]):
            return f"<vector len={len(obj)}>"
        # Large frame lists: show first + count.
        if len(obj) > 10:
            sample = [_sanitize_for_log(obj[0], _depth + 1)]
            return sample + [f"... ({len(obj)} items total)"]
        converted = [_sanitize_for_log(v, _depth + 1) for v in obj]
        return type(obj)(converted)
    if isinstance(obj, dict):
        return {k: _sanitize_for_log(v, _depth + 1) for k, v in obj.items()}
    return obj

def _get_request_timeout() -> float:
    """Return the per-request timeout in seconds (0 = no timeout)."""
    raw = os.environ.get("REQUEST_TIMEOUT_SECONDS", "0")
    try:
        return max(float(raw), 0.0)
    except (ValueError, TypeError):
        return 0.0

async def _await_with_timeout(future, timeout: float):
    """Await a future with an optional timeout. timeout <= 0 means no limit."""
    if timeout <= 0:
        return await future
    return await asyncio.wait_for(future, timeout=timeout)


def _aggregate_metrics(results, item_key):
    aggregate_metrics = {
        "preprocess_seconds": 0.0,
        "ai_inference_seconds": 0.0,
        item_key: 0,
    }
    for result in results:
        if not isinstance(result, dict):
            continue
        metrics = result.get("metrics") or {}
        if not metrics:
            continue
        aggregate_metrics["preprocess_seconds"] += metrics.get("preprocess_seconds", 0.0)
        aggregate_metrics["ai_inference_seconds"] += metrics.get("ai_inference_seconds", 0.0)
        aggregate_metrics[item_key] += 1
    aggregate_metrics["total_runtime_seconds"] = (
        aggregate_metrics["preprocess_seconds"] + aggregate_metrics["ai_inference_seconds"]
    )
    return aggregate_metrics


def _dedupe_names(names):
    seen = set()
    deduped = []
    for name in names or []:
        text = str(name or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _has_region_flow(pipeline_name):
    pm = server_manager.pipeline_manager
    if not pipeline_name or not pm.has_pipeline(pipeline_name):
        return False
    models = pm.dynamic_ai_manager.models
    capabilities = pm.dynamic_ai_manager.model_capabilities
    detector_names = capabilities.resolve_model_names_for_stage(pipeline_name, "detector", models) or []
    region_names = capabilities.resolve_model_names_for_stage(pipeline_name, "region", models) or []
    return bool(detector_names) and bool(region_names)


async def _resolve_face_recognition_model_names(pipeline_name):
    if not server_manager.pipeline_manager.has_pipeline(pipeline_name):
        await server_manager._ensure_pipelines_loaded(pipeline_name)
    if not _has_region_flow(pipeline_name):
        raise HTTPException(
            status_code=501,
            detail="Face recognition is not available. The required face recognition models are not active.",
        )

    pm = server_manager.pipeline_manager
    models = pm.dynamic_ai_manager.models
    capabilities = pm.dynamic_ai_manager.model_capabilities
    detector_names = capabilities.resolve_model_names_for_stage(pipeline_name, "detector", models) or []
    region_names = capabilities.resolve_model_names_for_stage(pipeline_name, "region", models) or []
    return _dedupe_names(detector_names + region_names)

logger = logging.getLogger("logger")

@app.post("/process_images/")
async def process_images(request: ImagePathList):
    try:
        image_paths = request.paths
        logger.info(f"Processing {len(image_paths)} images")
        pipeline_name = request.pipeline_name or server_manager.default_image_pipeline
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, request.return_confidence, None, None], pipeline_name,
            )
            for path in image_paths
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"error": str(result)}

        return_result = ImageResult(result=results)
        logger.debug(f"Returning Image Result: {_sanitize_for_log(return_result)}")
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
                data = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video, skipped_categories, None]

        try:
            timeout = _get_request_timeout()
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await _await_with_timeout(future, timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Video processing timed out")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return_result = VideoResult(result=result)
        logger.debug(f"Returning Video Result for: '{request.path}' Results: {_sanitize_for_log(return_result)}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/v3/process_video/")
async def process_video_v3(request: VideoRequestV3):
    try:
        logger.info(f"Processing video in v3 at path: {request.path}")

        pipeline_name = server_manager.default_video_pipeline
        
        data = [request.path, True, request.frame_interval, request.threshold, False, request.vr_video, request.categories_to_skip, None]

        result = None
        try:
            timeout = _get_request_timeout()
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await _await_with_timeout(future, timeout)
            logger.debug(f"Video v3 processing result: {_sanitize_for_log(result)}")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Video processing timed out")
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
        pipeline_name = server_manager.default_image_pipeline
        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        timeout = _get_request_timeout()
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, request.return_confidence, None, None], pipeline_name,
            )
            for path in image_paths
        ]
        if timeout > 0:
            results = await asyncio.gather(*[_await_with_timeout(f, timeout) for f in futures], return_exceptions=True)
        else:
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
        logger.debug(f"Returning Image Result v3: {_sanitize_for_log(return_payload)}")
        return return_payload
    except Exception as e:
        logger.error(f"Error processing images v3: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v3/face_recognition/process_images/")
async def process_images_face_recognition(request: ImageRequestV3):
    try:
        image_paths = request.paths
        logger.info(f"Processing {len(image_paths)} images (face recognition only)")
        pipeline_name = server_manager.default_image_pipeline
        requested_model_names = await _resolve_face_recognition_model_names(pipeline_name)
        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        timeout = _get_request_timeout()
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, request.return_confidence, None, requested_model_names], pipeline_name,
            )
            for path in image_paths
        ]
        if timeout > 0:
            results = await asyncio.gather(*[_await_with_timeout(f, timeout) for f in futures], return_exceptions=True)
        else:
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

        models = filter_pipeline_models(pipeline, requested_model_names)
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
        logger.debug(f"Returning Face Recognition Image Result: {_sanitize_for_log(return_payload)}")
        return return_payload
    except Exception as e:
        logger.error(f"Error processing images (face recognition): {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v3/face_recognition/process_video/")
async def process_video_face_recognition(request: VideoRequestV3):
    try:
        logger.info(f"Processing video (face recognition only) at path: {request.path}")

        pipeline_name = server_manager.default_video_pipeline
        requested_model_names = await _resolve_face_recognition_model_names(pipeline_name)

        data = [request.path, True, request.frame_interval, request.threshold, False, request.vr_video, request.categories_to_skip, requested_model_names]

        result = None
        try:
            timeout = _get_request_timeout()
            future = await server_manager.get_request_future(data, pipeline_name)
            result = await _await_with_timeout(future, timeout)
            logger.debug(f"Face recognition video result: {_sanitize_for_log(result)}")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Video processing timed out")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result

    except Exception as e:
        logger.error(f"Error processing video (face recognition): {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v3/current_ai_models/")
async def get_current_video_ai_models():
    try:
        pipeline_name = server_manager.default_video_pipeline
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


@app.get("/v3/capabilities/")
async def get_capabilities():
    """Return which high-level capabilities are currently available.

    Clients should call this on startup to discover what features are active
    (e.g. whether to show face-recognition UI).
    """
    try:
        pm = server_manager.pipeline_manager
        loaded = list(pm.pipelines.keys())

        capabilities = {
            "face_recognition": (
                _has_region_flow(server_manager.default_image_pipeline)
                or _has_region_flow(server_manager.default_video_pipeline)
            ),
            "image_tagging": pm.has_pipeline(server_manager.default_image_pipeline),
            "video_tagging": pm.has_pipeline(server_manager.default_video_pipeline),
            "visual_embeddings": any(
                "embedding" in set(getattr(m.model, "model_capabilities", []) or [])
                for m in pm.model_manager.models.values()
                if m is not None and hasattr(m, "model")
            ),
            "loaded_pipelines": loaded,
        }
        return capabilities
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v4/models/catalog")
async def get_v4_model_catalog():
    try:
        return {"models": get_model_catalog(server_manager)}
    except Exception as e:
        logger.error(f"Error getting v4 model catalog: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v4/capabilities")
async def get_v4_capabilities():
    try:
        return {
            "capabilities": get_capability_catalog(server_manager),
            "load_policies": sorted(VALID_LOAD_POLICIES),
        }
    except Exception as e:
        logger.error(f"Error getting v4 capabilities: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v4/health")
async def get_v4_health():
    try:
        catalog = get_model_catalog(server_manager)
        loaded_models = get_loaded_models(server_manager)
        pipelines = server_manager.pipeline_manager.pipelines

        response = {
            "status": "ok",
            "time": time.time(),
            "loaded_pipelines": sorted(pipelines.keys()),
            "loaded_pipeline_count": len(pipelines),
            "model_count": len(catalog),
            "loaded_model_count": len(loaded_models),
            "active_model_count": sum(1 for entry in catalog if entry.get("active")),
            "request_timeout_seconds": _get_request_timeout(),
        }

        try:
            response["outstanding_requests"] = outstanding_requests_middleware.outstanding_requests
        except Exception:
            response["outstanding_requests"] = None

        try:
            response["cuda_available"] = torch.cuda.is_available()
            if response["cuda_available"]:
                response["cuda_device_count"] = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                response["current_device"] = current_device
                response["current_device_name"] = torch.cuda.get_device_name(current_device)
                response["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated()
                response["cuda_memory_reserved_bytes"] = torch.cuda.memory_reserved()
        except Exception as e:
            response["cuda_check_error"] = str(e)

        return response
    except Exception as e:
        logger.error(f"Error getting v4 health: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v4/models/loaded")
async def get_v4_loaded_models():
    try:
        return {"models": get_loaded_models(server_manager)}
    except Exception as e:
        logger.error(f"Error getting loaded v4 models: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/models/load")
async def load_v4_models(request: ModelSelectionRequestV4):
    try:
        loaded_models = await load_models(server_manager, request.models)
        return {"models": loaded_models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading v4 models: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/models/unload")
async def unload_v4_models(request: ModelSelectionRequestV4):
    try:
        loaded_models = await unload_models(server_manager, request.models)
        return {"models": loaded_models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error unloading v4 models: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/encode/text")
async def encode_v4_text(request: TextEncodeRequestV4):
    try:
        return await asyncio.to_thread(text_encoding_service.encode, request.kind_family, request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error encoding text v4: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/pipelines/custom")
async def register_v4_custom_pipeline(request: CustomPipelineRequestV4):
    try:
        return await server_manager.register_custom_pipeline(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering custom v4 pipeline: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v4/pipelines/custom/{pipeline_name}")
async def delete_v4_custom_pipeline(pipeline_name: str):
    try:
        return await server_manager.delete_custom_pipeline(pipeline_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting custom v4 pipeline: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/analyze/images")
async def process_images_v4(request: ImageRequestV4):
    try:
        requested_model_names, reloaded = await resolve_request_models(
            server_manager,
            request.want,
            default_scope="asset",
            load_policy=request.load_policy,
        )
        pipeline_name = request.pipeline_name or server_manager.default_image_pipeline
        requested_model_names = expand_requested_models_with_dependencies(
            server_manager, pipeline_name, requested_model_names
        )
        timeout = _get_request_timeout()
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, request.return_confidence, request.categories_to_skip, requested_model_names or None],
                pipeline_name,
            )
            for path in request.paths
        ]
        if timeout > 0:
            results = await asyncio.gather(*[_await_with_timeout(f, timeout) for f in futures], return_exceptions=True)
        else:
            results = await asyncio.gather(*futures, return_exceptions=True)

        for index, result in enumerate(results):
            if isinstance(result, Exception):
                results[index] = {"error": str(result)}

        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        return {
            "result": results,
            "requested_model_names": requested_model_names,
            "models": filter_pipeline_models(pipeline, requested_model_names),
            "reloaded": reloaded,
            "metrics": _aggregate_metrics(results, "image_count"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing images v4: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/analyze/video")
async def process_video_v4(request: VideoRequestV4):
    try:
        requested_model_names, reloaded = await resolve_request_models(
            server_manager,
            request.want,
            default_scope="frame",
            load_policy=request.load_policy,
        )
        pipeline_name = request.pipeline_name or server_manager.default_video_pipeline
        requested_model_names = expand_requested_models_with_dependencies(
            server_manager, pipeline_name, requested_model_names
        )
        timeout = _get_request_timeout()
        future = await server_manager.get_request_future(
            [
                request.path,
                True,
                request.frame_interval,
                request.threshold,
                request.return_confidence,
                request.vr_video,
                request.categories_to_skip,
                requested_model_names or None,
            ],
            pipeline_name,
        )
        result = await _await_with_timeout(future, timeout)
        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        # === TEMP FACE_DEBUG: explain why each ai model ran or was skipped ===
        try:
            _dbg = logging.getLogger("logger")
            _skip_cats = set(request.categories_to_skip or [])
            _requested = {str(n).strip() for n in (requested_model_names or []) if str(n).strip()}
            _dbg.info(f"FACE_DEBUG path={request.path}")
            _dbg.info(f"FACE_DEBUG categories_to_skip={sorted(_skip_cats)}")
            _dbg.info(f"FACE_DEBUG requested_model_names={sorted(_requested)}")
            for _i, _w in enumerate(request.want or []):
                _dbg.info(
                    f"FACE_DEBUG want[{_i}] capability={_w.capability} capabilities={_w.capabilities} "
                    f"scope={_w.scope} from_detection={_w.from_detection} models={_w.models}"
                )
            for _mi in pipeline.get_ai_models_info():
                _cats = list(_mi.categories or [])
                _names = {n for n in {str(_mi.config_name or "").strip(), str(_mi.name or "").strip()} if n}
                _cat_skip = bool(_cats) and all(_c in _skip_cats for _c in _cats)
                _name_skip = bool(_requested) and _names.isdisjoint(_requested)
                if _cat_skip:
                    _why = "SKIPPED (all categories in categories_to_skip)"
                elif _name_skip:
                    _why = "SKIPPED (config_name/model_file_name not in requested_model_names)"
                else:
                    _why = "RUNS"
                _dbg.info(
                    f"FACE_DEBUG model config_name={_mi.config_name} file_name={_mi.name} "
                    f"categories={_cats} -> {_why}"
                )
        except Exception as _dbg_exc:
            logging.getLogger("logger").warning(f"FACE_DEBUG failed: {_dbg_exc}")
        # === END TEMP FACE_DEBUG ===
        result["requested_model_names"] = requested_model_names
        result["models"] = filter_pipeline_models(pipeline, requested_model_names)
        result["reloaded"] = reloaded
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Video processing timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing video v4: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v4/analyze/audio")
async def process_audio_v4(request: AudioRequestV4):
    try:
        requested_model_names, reloaded = await resolve_request_models(
            server_manager,
            request.want,
            default_scope="asset",
            load_policy=request.load_policy,
        )
        pipeline_name = request.pipeline_name or server_manager.default_audio_pipeline
        if not server_manager.pipeline_manager.has_pipeline(pipeline_name):
            raise HTTPException(
                status_code=501,
                detail="Audio v4 pipeline is not available. The required audio models are not active.",
            )
        timeout = _get_request_timeout()
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, requested_model_names or None],
                pipeline_name,
            )
            for path in request.paths
        ]
        if timeout > 0:
            results = await asyncio.gather(*[_await_with_timeout(f, timeout) for f in futures], return_exceptions=True)
        else:
            results = await asyncio.gather(*futures, return_exceptions=True)

        for index, result in enumerate(results):
            if isinstance(result, Exception):
                results[index] = {"error": str(result)}

        pipeline = server_manager.pipeline_manager.get_pipeline(pipeline_name)
        return {
            "result": results,
            "requested_model_names": requested_model_names,
            "models": filter_pipeline_models(pipeline, requested_model_names),
            "reloaded": reloaded,
            "metrics": _aggregate_metrics(results, "audio_count"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing audio v4: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/process_audio/")
async def process_audio(request: AudioRequest):
    """Process audio files through the audio embedding pipeline.

    Extracts audio from media files and produces audio embeddings
    and optional classification scores for gating/type-binning.
    """
    try:
        audio_paths = request.paths
        logger.info(f"Processing {len(audio_paths)} audio files")
        pipeline_name = server_manager.default_audio_pipeline
        if not server_manager.pipeline_manager.has_pipeline(pipeline_name):
            raise HTTPException(
                status_code=501,
                detail="Audio pipeline is not available. The required audio models are not active.",
            )
        timeout = _get_request_timeout()
        futures = [
            await server_manager.get_request_future(
                [path, request.threshold, None], pipeline_name
            )
            for path in audio_paths
        ]
        if timeout > 0:
            results = await asyncio.gather(
                *[_await_with_timeout(f, timeout) for f in futures],
                return_exceptions=True,
            )
        else:
            results = await asyncio.gather(*futures, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"error": str(result)}

        return_payload = {
            "result": results,
        }
        logger.debug(f"Returning Audio Result: {_sanitize_for_log(return_payload)}")
        return return_payload
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
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