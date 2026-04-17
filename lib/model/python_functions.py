
import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.preprocessing_python.image_preprocessing import get_video_duration_av
from lib.model.postprocessing.AI_VideoResult import AIVideoResult, AIVideoResultV3
import lib.model.postprocessing.timeframe_processing as timeframe_processing
from lib.model.postprocessing.category_settings import category_config
from lib.model.skip_input import Skip
from lib.model.postprocessing.post_processing_settings import get_or_default, post_processing_config
from lib.pipeline.target_primitives import build_region_target, targets_to_dicts

logger = logging.getLogger("logger")

async def result_coalescer(data):
    for item in data:
        itemFuture = item.item_future
        result = {}
        structured_outputs = []
        structured_errors = []
        for input_name in item.input_names:
            ai_result = itemFuture[input_name]
            if isinstance(ai_result, Skip):
                structured_outputs.append(
                    {
                        "model_step": input_name,
                        "status": "skipped",
                    }
                )
                continue

            if isinstance(ai_result, Exception):
                error_text = str(ai_result)
                structured_outputs.append(
                    {
                        "model_step": input_name,
                        "status": "error",
                        "error": error_text,
                    }
                )
                structured_errors.append({"model_step": input_name, "error": error_text})
                continue

            result[input_name] = ai_result
            structured_outputs.append(
                {
                    "model_step": input_name,
                    "status": "ok",
                }
            )

        result["_outputs"] = structured_outputs
        if structured_errors:
            result["_errors"] = structured_errors
        await itemFuture.set_data(item.output_names[0], result)
        
async def result_finisher(data):
    for item in data:
        itemFuture = item.item_future
        future_results = itemFuture[item.input_names[0]]
        itemFuture.close_future(future_results)

async def batch_awaiter(data):
    """
    Non-blocking child-future joiner.

    Instead of blocking the worker while waiting for children to complete,
    this spawns a lightweight async task per item and returns immediately.
    The worker is freed to process other queued items. When all children
    finish, the task writes the aggregated result back via set_data,
    which re-enters the pipeline event handler normally.

    This means batch_awaiter can safely be nested (e.g. video frame children
    that themselves have region children) without deadlocking, because no
    worker is held while waiting.
    """
    for item in data:
        itemFuture = item.item_future
        child_futures = itemFuture[item.input_names[0]]
        if child_futures is None:
            child_futures = []
        if not isinstance(child_futures, list):
            child_futures = [child_futures]

        # Fire-and-forget: spawn a task that monitors children and writes
        # the result back to the parent future when they all complete.
        # The worker returns immediately after this loop.
        asyncio.create_task(
            _join_child_futures(itemFuture, child_futures, item.output_names[0])
        )


async def _join_child_futures(parent_future, child_futures, output_name):
    """Await all child futures and write aggregated results to the parent."""
    try:
        # Collect the underlying asyncio.Future from each ItemFuture
        raw_futures = []
        for child in child_futures:
            if hasattr(child, "future") and child.future is not None:
                raw_futures.append(child.future)
            elif isinstance(child, asyncio.Future):
                raw_futures.append(child)
            else:
                raw_futures.append(asyncio.ensure_future(child))

        if not raw_futures:
            await parent_future.set_data(output_name, [])
            return

        results = await asyncio.gather(*raw_futures, return_exceptions=True)

        normalized_results = []
        for result in results:
            if isinstance(result, Exception):
                normalized_results.append({"_error": str(result), "_status": "error"})
            elif isinstance(result, dict):
                # Strip pipeline-internal keys (_outputs, _errors) from child
                # results before they become part of the parent response.
                normalized_results.append(
                    {k: v for k, v in result.items() if not isinstance(k, str) or not k.startswith("_")}
                )
            else:
                normalized_results.append(result)

        await parent_future.set_data(output_name, normalized_results)
    except Exception as e:
        logger.error(f"_join_child_futures failed: {e}", exc_info=True)
        try:
            parent_future.set_exception(e)
        except Exception:
            pass

async def video_result_postprocessor(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_av(itemFuture[item.input_names[1]])
        result = {"frames": itemFuture[item.input_names[0]], "video_duration": duration, "frame_interval": float(itemFuture[item.input_names[2]]), "threshold": float(itemFuture[item.input_names[3]]), "ai_models_info": itemFuture['pipeline'].get_ai_models_info()}
        del itemFuture.data["pipeline"]

        videoResult = itemFuture[item.input_names[4]]
        if videoResult is not None:
            videoResult.add_server_result(result)
        else:
            videoResult = AIVideoResult.from_server_result(result)

        toReturn = {"json_result": videoResult.to_json(), "video_tag_info": timeframe_processing.compute_video_tag_info(videoResult)}
        
        await itemFuture.set_data(item.output_names[0], toReturn)

async def video_result_postprocessor_v3(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_av(itemFuture[item.input_names[1]])
        pipeline = itemFuture['pipeline']
        currently_active_models = pipeline.get_ai_models_info()

        raw_frames = itemFuture[item.input_names[0]]
        clean_frames, _, _ = _extract_structured_frames(raw_frames)
        timespan_frames = _filter_timespan_compatible_frames(clean_frames)
        per_frame_data = _extract_per_frame_data(clean_frames)

        raw_frame_interval = itemFuture[item.input_names[2]]
        raw_threshold = itemFuture[item.input_names[3]]

        result = {
            "frames": timespan_frames,
            "per_frame_data": per_frame_data,
            "video_duration": duration,
            "frame_interval": float(raw_frame_interval) if raw_frame_interval is not None else 0.5,
            "threshold": float(raw_threshold) if raw_threshold is not None else 0.5,
            "ai_models_info": currently_active_models,
            "skipped_categories": itemFuture[item.input_names[4]],
        }

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        used_models = [
            model for model in currently_active_models
            if not all(
                category in (result.get("skipped_categories") or [])
                for category in model.categories
            )
        ]
        result["models"] = used_models
        max_merge_seconds = post_processing_config.get("max_timespan_merge_seconds", 2)

        try:
            videoResult = AIVideoResultV3.from_server_result(result, max_merge_seconds=max_merge_seconds)
        except Exception as e:
            logger.error(f"Error creating AIVideoResultV3: {e}")
            raise

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)
        metrics.setdefault("preprocess_seconds", 0.0)
        metrics.setdefault("ai_inference_seconds", 0.0)
        metrics["ai_model_count"] = len(currently_active_models)

        total_runtime = metrics.get("preprocess_seconds", 0.0) + metrics.get("ai_inference_seconds", 0.0)
        metrics["total_runtime_seconds"] = total_runtime

        payload = {"result": videoResult, "metrics": metrics}

        await itemFuture.set_data(item.output_names[0], payload)

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
                    
                    tag_threshold = float(get_or_default(category_config[category][tagname], 'TagThreshold', 0.5))
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

async def image_result_postprocessor_v3(data):
    for item in data:
        itemFuture = item.item_future
        pipeline = itemFuture['pipeline']
        result = itemFuture[item.input_names[0]]
        coalesced_outputs = []
        coalesced_errors = []
        if isinstance(result, dict):
            coalesced_outputs = list(result.get("_outputs") or [])
            coalesced_errors = list(result.get("_errors") or [])
            result = {
                key: value
                for key, value in result.items()
                if not str(key).startswith("_")
            }
        # v3 API: do not rely on category_config for filtering/renaming.
        # Preserve the model output shape (category -> list of tags/tuples) as-is.
        toReturn = {}
        if isinstance(result, dict):
            for category, tags in result.items():
                if tags is None:
                    toReturn[category] = []
                    continue
                if isinstance(tags, (list, tuple)):
                    toReturn[category] = list(tags)
                else:
                    # Defensive: if a backend returns a single tag, wrap it.
                    toReturn[category] = [tags]
        else:
            # Defensive: unexpected shape, return as-is.
            toReturn = result

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)
        metrics.setdefault("preprocess_seconds", 0.0)
        metrics.setdefault("ai_inference_seconds", 0.0)
        metrics["ai_model_count"] = len(pipeline.get_ai_models_info())

        total_runtime = metrics.get("preprocess_seconds", 0.0) + metrics.get("ai_inference_seconds", 0.0)
        metrics["total_runtime_seconds"] = total_runtime

        source_asset_id = itemFuture["image_path"]
        target = {
            "target_id": "asset:0",
            "scope": "asset",
            "source_asset_id": str(source_asset_id) if source_asset_id is not None else "unknown",
        }
        results_v2 = {
            "assets": [
                {
                    "asset_id": target["source_asset_id"],
                    "targets": [target],
                    "outputs": coalesced_outputs,
                    "errors": coalesced_errors,
                }
            ],
            "errors": coalesced_errors,
        }

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        payload = {"result": toReturn, "results_v2": results_v2, "metrics": metrics}

        await itemFuture.set_data(item.output_names[0], payload)


async def audio_result_postprocessor(data):
    """Post-processor for the audio embedding pipeline.

    Receives per-window collated results from batch_awaiter, then:
      1. AST semantic filtering — reject windows dominated by music
      2. Type-binning — classify kept windows as moan/speech/breath
      3. Per-type centroid embedding — mean of ECAPA vectors per type, L2-normalised
      4. Overall centroid — mean of all kept ECAPA vectors
    """
    for item in data:
        itemFuture = item.item_future
        children_results = itemFuture[item.input_names[0]]  # list of per-window dicts
        audio_path = itemFuture[item.input_names[1]] if len(item.input_names) > 1 else "unknown"

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)

        if not isinstance(children_results, list):
            children_results = []

        # ── AudioSet category indices ──
        # Whitelist: human non-speech vocalizations we want to keep
        IDX_WHITELIST = [8, 9, 14, 22, 24, 25, 38, 39, 44, 45, 46]
        # Speech
        IDX_SPEECH = [0, 1, 2, 3, 4, 5, 15]
        # Music (reject if dominant)
        IDX_MUSIC = [
            27, 28, 32, 33, 34, 137, 138, 140, 141, 142, 143, 144, 145, 146,
            153, 154, 162, 163, 164, 165, 166, 167, 168, 184, 266, 270,
        ]
        # Type bins for kept windows
        IDX_MOAN = [25, 38, 39, 22, 24, 44, 45, 46, 14, 8, 9]
        IDX_BREATH = [41, 26, 43]

        threshold = 0.01  # AST sigmoid scores are typically low for specific classes

        kept_windows = []
        rejected_music = 0
        rejected_quiet = 0

        for win in children_results:
            if not isinstance(win, dict):
                continue
            if "_error" in win:
                continue

            # Extract AST probabilities
            ast_list = win.get("audio_classification_ast", [])
            probs = None
            if ast_list and isinstance(ast_list, list) and isinstance(ast_list[0], dict):
                probs = ast_list[0].get("probabilities")

            if probs is None:
                # No classification available — keep by default
                kept_windows.append(win)
                continue

            # Compute category scores
            music_score = max((probs[i] for i in IDX_MUSIC if i < len(probs)), default=0.0)
            whitelist_score = max((probs[i] for i in IDX_WHITELIST if i < len(probs)), default=0.0)
            speech_score = max((probs[i] for i in IDX_SPEECH if i < len(probs)), default=0.0)

            # Reject music-dominated windows
            if music_score > 0.05 and music_score > max(whitelist_score, speech_score):
                rejected_music += 1
                continue

            # Keep if any relevant vocal activity
            keep_score = max(whitelist_score, speech_score)
            if keep_score < threshold:
                rejected_quiet += 1
                continue

            # Annotate window with scores for downstream use
            win["_scores"] = {
                "whitelist": round(whitelist_score, 4),
                "speech": round(speech_score, 4),
                "music": round(music_score, 4),
            }
            kept_windows.append(win)

        # ── Type-binning ──
        type_bins = {"moan": [], "speech": [], "breath": []}

        for win in kept_windows:
            probs = None
            ast_list = win.get("audio_classification_ast", [])
            if ast_list and isinstance(ast_list, list) and isinstance(ast_list[0], dict):
                probs = ast_list[0].get("probabilities")

            # Extract embedding vector
            emb_list = win.get("audio_embeddings_ecapa", [])
            vector = None
            if emb_list and isinstance(emb_list, list) and isinstance(emb_list[0], dict):
                vector = emb_list[0].get("vector")

            if vector is None:
                continue

            if probs is not None:
                moan_score = max((probs[i] for i in IDX_MOAN if i < len(probs)), default=0.0)
                speech_score = max((probs[i] for i in IDX_SPEECH if i < len(probs)), default=0.0)
                breath_score = max((probs[i] for i in IDX_BREATH if i < len(probs)), default=0.0)

                scores = {"moan": moan_score, "speech": speech_score, "breath": breath_score}
                dominant = max(scores, key=scores.get)
                win["_dominant_type"] = dominant
            else:
                dominant = "moan"  # default bin
                win["_dominant_type"] = dominant

            type_bins[dominant].append(vector)

        # ── Per-type centroid embeddings ──
        embeddings = {}
        for type_name, vectors in type_bins.items():
            if not vectors:
                continue
            arr = np.array(vectors, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid = centroid / norm
            embeddings[type_name] = {
                "centroid": centroid.tolist(),
                "norm": round(float(np.linalg.norm(centroid)), 6),
                "dim": int(centroid.shape[0]),
                "window_count": len(vectors),
            }

        # ── Overall centroid ──
        all_vectors = [v for vecs in type_bins.values() for v in vecs]
        overall_centroid = None
        if all_vectors:
            arr = np.array(all_vectors, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid = centroid / norm
            overall_centroid = {
                "centroid": centroid.tolist(),
                "norm": round(float(np.linalg.norm(centroid)), 6),
                "dim": int(centroid.shape[0]),
                "window_count": len(all_vectors),
            }

        # ── Duration computation (merge overlapping windows per type) ──
        def _merge_intervals(intervals):
            """Merge overlapping/adjacent (start, end) intervals, return total seconds."""
            if not intervals:
                return 0.0, []
            sorted_iv = sorted(intervals)
            merged = [sorted_iv[0]]
            for s, e in sorted_iv[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            total = sum(e - s for s, e in merged)
            return total, merged

        type_windows = {"moan": [], "speech": [], "breath": []}
        all_kept_intervals = []
        for win in kept_windows:
            start = win.get("window_start")
            end = win.get("window_end")
            if start is None or end is None:
                continue
            dominant = win.get("_dominant_type", "unknown")
            if dominant in type_windows:
                type_windows[dominant].append((start, end))
            all_kept_intervals.append((start, end))

        duration_by_type = {}
        for type_name, intervals in type_windows.items():
            secs, _ = _merge_intervals(intervals)
            if secs > 0:
                duration_by_type[type_name] = round(secs, 2)

        total_kept_duration, _ = _merge_intervals(all_kept_intervals)

        # ── Build response ──
        metrics["windows_total"] = len(children_results)
        metrics["windows_kept"] = len(kept_windows)
        metrics["windows_rejected_music"] = rejected_music
        metrics["windows_rejected_quiet"] = rejected_quiet
        metrics["type_counts"] = {k: len(v) for k, v in type_bins.items()}

        if hasattr(itemFuture, "data") and isinstance(itemFuture.data, dict):
            itemFuture.data.pop("pipeline", None)

        payload = {
            "embeddings": embeddings,
            "overall_embedding": overall_centroid,
            "classification_summary": {
                "windows_analyzed": len(children_results),
                "windows_kept": len(kept_windows),
                "rejected_music": rejected_music,
                "rejected_quiet": rejected_quiet,
                "type_distribution": {k: len(v) for k, v in type_bins.items()},
                "duration_seconds": duration_by_type,
                "total_kept_duration_seconds": round(total_kept_duration, 2),
            },
            "windows": [
                {
                    "index": w.get("window_index"),
                    "start": w.get("window_start"),
                    "end": w.get("window_end"),
                    "dominant_type": w.get("_dominant_type", "unknown"),
                    "scores": w.get("_scores", {}),
                }
                for w in kept_windows
            ],
            "metrics": metrics,
            "source": str(audio_path),
        }

        await itemFuture.set_data(item.output_names[0], payload)


async def detector_result_to_region_targets(data):
    max_targets_raw = os.environ.get("MAX_REGION_TARGETS_PER_ITEM", "64")
    try:
        max_targets_per_item = int(max_targets_raw)
    except Exception:
        max_targets_per_item = 64

    for item in data:
        itemFuture = item.item_future
        detections = itemFuture[item.input_names[0]]
        source_asset_id = itemFuture[item.input_names[1]]

        frame_index = None
        source_tensor = None
        parent_target_id = None
        bbox_scale_x = 1.0
        bbox_scale_y = 1.0

        if len(item.input_names) > 2:
            third_input = itemFuture[item.input_names[2]]
            if isinstance(third_input, torch.Tensor):
                source_tensor = third_input
            else:
                frame_index = third_input

        if len(item.input_names) > 3:
            fourth_input = itemFuture[item.input_names[3]]
            if source_tensor is None and isinstance(fourth_input, torch.Tensor):
                source_tensor = fourth_input
            elif frame_index is None:
                frame_index = fourth_input
            else:
                parent_target_id = fourth_input

        if len(item.input_names) > 4:
            fifth_input = itemFuture[item.input_names[4]]
            if not isinstance(fifth_input, torch.Tensor):
                parent_target_id = fifth_input

        source_height, source_width = _extract_tensor_hw(source_tensor)

        # Compute bbox scale AFTER source dimensions are known.
        if len(item.input_names) > 4:
            fifth_input = itemFuture[item.input_names[4]]
            if isinstance(fifth_input, torch.Tensor):
                det_h, det_w = _extract_tensor_hw(fifth_input)
                if (det_h and det_w and source_height and source_width
                        and (det_h != source_height or det_w != source_width)):
                    bbox_scale_x = source_width / det_w
                    bbox_scale_y = source_height / det_h

        candidate_detections = _extract_detection_items(detections)

        if max_targets_per_item > 0 and len(candidate_detections) > max_targets_per_item:
            sortable = []
            unsorted = []
            for det in candidate_detections:
                if isinstance(det, dict) and isinstance(det.get("score", None), (int, float)):
                    sortable.append(det)
                else:
                    unsorted.append(det)
            sortable.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            candidate_detections = (sortable + unsorted)[:max_targets_per_item]
            logger.warning(
                "Capped region targets to %s for source '%s'",
                max_targets_per_item,
                source_asset_id,
            )

        region_targets = []
        region_errors = []
        for detection_index, detection in enumerate(candidate_detections):
            bbox = _extract_detection_bbox(detection)
            if bbox is None:
                region_errors.append(
                    {
                        "index": detection_index,
                        "error": "missing_bbox",
                        "raw_detection": detection,
                    }
                )
                continue

            # Scale bbox from detector-input space to region-source space
            if bbox_scale_x != 1.0 or bbox_scale_y != 1.0:
                bbox = [bbox[0] * bbox_scale_x, bbox[1] * bbox_scale_y,
                        bbox[2] * bbox_scale_x, bbox[3] * bbox_scale_y]

            # Scale kps similarly
            kps_raw = detection.get("kps") if isinstance(detection, dict) else None
            if kps_raw and (bbox_scale_x != 1.0 or bbox_scale_y != 1.0):
                kps_raw = [[x * bbox_scale_x, y * bbox_scale_y] for x, y in kps_raw]

            try:
                target = build_region_target(
                    source_asset_id=str(source_asset_id),
                    bbox=bbox,
                    frame_index=frame_index,
                    parent_target_id=parent_target_id,
                    source_width=source_width,
                    source_height=source_height,
                    metadata={
                        "detection_index": detection_index,
                        "kps": kps_raw,
                    },
                )
                region_targets.append(target)
            except Exception as exc:
                region_errors.append(
                    {
                        "index": detection_index,
                        "error": str(exc),
                        "raw_detection": detection,
                    }
                )

        # Write client-safe detections (without kps, with normalized bbox) back
        # to the detection key.  Downstream steps (region_children_builder,
        # embedding models) already extracted kps/bbox into region-target
        # metadata, so the original value is no longer needed.  Direct
        # mutation avoids re-triggering the pipeline event handler — safe
        # because the detection key has already been consumed by this step.
        if itemFuture.data is not None:
            can_normalize = (source_width is not None and source_height is not None
                             and source_width > 0 and source_height > 0)
            client_detections = []
            for det in candidate_detections:
                if isinstance(det, dict):
                    client_det = {k: v for k, v in det.items() if k != "kps"}
                    if can_normalize and "bbox" in client_det:
                        bx1, by1, bx2, by2 = client_det["bbox"]
                        # Scale from detector-input space to region-source
                        # space before normalising to 0–1.
                        bx1 *= bbox_scale_x; by1 *= bbox_scale_y
                        bx2 *= bbox_scale_x; by2 *= bbox_scale_y
                        client_det["bbox"] = [
                            max(0.0, min(1.0, bx1 / source_width)),
                            max(0.0, min(1.0, by1 / source_height)),
                            max(0.0, min(1.0, bx2 / source_width)),
                            max(0.0, min(1.0, by2 / source_height)),
                        ]
                    client_detections.append(client_det)
                else:
                    client_detections.append(det)
            itemFuture.data[item.input_names[0]] = client_detections

        await itemFuture.set_data(item.output_names[0], targets_to_dicts(region_targets))
        if len(item.output_names) > 1:
            await itemFuture.set_data(item.output_names[1], region_errors)


async def region_children_builder(data):
    for item in data:
        itemFuture = item.item_future
        source_tensor = itemFuture[item.input_names[0]]
        region_targets = itemFuture[item.input_names[1]] or []
        threshold = itemFuture[item.input_names[2]] if len(item.input_names) > 2 else None
        return_confidence = itemFuture[item.input_names[3]] if len(item.input_names) > 3 else None
        skipped_categories = itemFuture[item.input_names[4]] if len(item.input_names) > 4 else None

        children = []
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError("region_children_builder requires source_tensor input as a torch.Tensor")

        source_height, source_width = _extract_tensor_hw(source_tensor)
        if source_height is None or source_width is None:
            raise ValueError("region_children_builder could not determine source tensor dimensions")

        for region_target in region_targets:
            bbox = region_target.get("bbox") if isinstance(region_target, dict) else None
            if bbox is None:
                continue

            # Extract detection_index for correlation; the full region
            # target stays available for models that need it (e.g. kps
            # for face alignment) but won't be coalesced into the result.
            det_index = None
            if isinstance(region_target, dict):
                meta = region_target.get("metadata") or {}
                det_index = meta.get("detection_index")

            crop_tensor = _crop_and_resize_region(source_tensor, bbox)
            payload = {
                item.output_names[1]: crop_tensor,
                item.output_names[2]: region_target,
                item.output_names[3]: threshold,
                item.output_names[4]: return_confidence,
                item.output_names[5]: skipped_categories,
            }
            if len(item.output_names) > 6:
                payload[item.output_names[6]] = source_tensor
            if len(item.output_names) > 7:
                payload[item.output_names[7]] = det_index
            child_future = await ItemFuture.create(item, payload, item.item_future.handler)
            children.append(child_future)

        # Release the parent frame's reference to the full-resolution
        # source tensor.  Children that need it (face alignment) already
        # hold their own reference.  Clearing it here allows GC to reclaim
        # the large tensor as soon as all children finish with it.
        if itemFuture.data is not None:
            src_key = item.input_names[0]
            if src_key in itemFuture.data:
                itemFuture.data[src_key] = None

        await itemFuture.set_data(item.output_names[0], children)


def _extract_tensor_hw(source_tensor):
    if isinstance(source_tensor, torch.Tensor):
        if source_tensor.dim() == 3:
            return int(source_tensor.shape[-2]), int(source_tensor.shape[-1])
        if source_tensor.dim() == 4:
            return int(source_tensor.shape[-2]), int(source_tensor.shape[-1])
    return None, None


def _extract_detection_items(detections):
    if detections is None:
        return []
    if isinstance(detections, dict):
        if "detections" in detections and isinstance(detections["detections"], list):
            return detections["detections"]
        for value in detections.values():
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict) and ("bbox" in first or "box" in first):
                    return value
        return [detections]
    if isinstance(detections, list):
        return detections
    return []


def _extract_detection_bbox(detection):
    if detection is None:
        return None
    if isinstance(detection, dict):
        bbox = detection.get("bbox", None)
        if bbox is not None:
            return bbox
        alt = detection.get("box", None)
        if alt is not None:
            return alt
    if isinstance(detection, (list, tuple)) and len(detection) == 4:
        return detection
    return None


def _crop_and_resize_region(source_tensor: torch.Tensor, bbox: Sequence[float]) -> torch.Tensor:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]

    tensor = source_tensor
    squeeze = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True

    source_h = int(tensor.shape[-2])
    source_w = int(tensor.shape[-1])

    x1 = min(max(x1, 0), source_w)
    x2 = min(max(x2, 0), source_w)
    y1 = min(max(y1, 0), source_h)
    y2 = min(max(y2, 0), source_h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox after clamping; zero-area crop")

    # .clone() to own memory independently of the source tensor so
    # the full-resolution source can be garbage-collected promptly.
    cropped = tensor[..., y1:y2, x1:x2].clone()
    if squeeze:
        cropped = cropped.squeeze(0)
    return cropped


def _round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _extract_structured_frames(raw_frames):
    clean_frames = []
    frame_structured_outputs = []
    frame_errors = []

    if raw_frames is None:
        return clean_frames, frame_structured_outputs, frame_errors

    for frame_index, frame in enumerate(raw_frames):
        if isinstance(frame, Exception):
            frame_errors.append(
                {
                    "frame": frame_index,
                    "error": str(frame),
                    "status": "error",
                }
            )
            continue

        if not isinstance(frame, dict):
            frame_errors.append(
                {
                    "frame": frame_index,
                    "error": f"Unexpected frame payload type: {type(frame).__name__}",
                    "status": "error",
                }
            )
            continue

        frame_outputs = frame.get("_outputs") or []
        for output in frame_outputs:
            output_copy = dict(output)
            output_copy["frame"] = frame.get("frame_index", frame_index)
            frame_structured_outputs.append(output_copy)

        for error in frame.get("_errors") or []:
            error_copy = dict(error)
            error_copy["frame"] = frame.get("frame_index", frame_index)
            frame_errors.append(error_copy)

        clean_frame = {key: value for key, value in frame.items() if not str(key).startswith("_")}
        if "frame_index" not in clean_frame:
            clean_frame["frame_index"] = frame_index
        clean_frames.append(clean_frame)

    return clean_frames, frame_structured_outputs, frame_errors


def _is_tag_list(value):
    """Return True if *value* looks like a tagging model output (list of
    strings or (string, float) tuples) suitable for timespan building.
    An empty list is considered a valid (empty) tag list."""
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str):
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            continue
        return False
    return True


def _filter_timespan_compatible_frames(frames):
    filtered_frames = []
    for frame_index, frame in enumerate(frames or []):
        if not isinstance(frame, dict):
            continue

        filtered = {"frame_index": frame.get("frame_index", frame_index)}
        for key, value in frame.items():
            if key == "frame_index":
                continue
            if value and _is_tag_list(value):
                filtered[key] = value

        filtered_frames.append(filtered)

    return filtered_frames


def _extract_per_frame_data(clean_frames):
    """Extract non-tag per-frame data (detections, regions, etc.).

    Returns a list of dicts, each containing ``frame_index`` plus any
    keys whose values are *not* tag lists (i.e. structured data like
    detection dicts or region result lists).  Frames with no such data
    are omitted to keep the payload compact."""
    per_frame = []
    for frame_index, frame in enumerate(clean_frames or []):
        if not isinstance(frame, dict):
            continue

        entry = {"frame_index": frame.get("frame_index", frame_index)}
        for key, value in frame.items():
            if key == "frame_index":
                continue
            # Skip tag lists (handled by timespans), None, and empty values.
            if not value:
                continue
            if _is_tag_list(value):
                continue
            entry[key] = value

        # Only include if there's actual structured data beyond frame_index.
        if len(entry) > 1:
            per_frame.append(entry)

    return per_frame
