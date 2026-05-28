
import logging
from lib.async_lib.async_processing import ItemFuture
from lib.config.config_utils import load_config
from lib.pipeline.dynamic_ai_manager import DynamicAIManager
from lib.pipeline.pipeline import Pipeline
from lib.model.model_manager import ModelManager
from lib.server.exceptions import NoActiveModelsException, ServerStopException

class PipelineManager:
    def __init__(self):
        self.pipelines = {}
        self.logger = logging.getLogger("logger")
        self.model_manager = ModelManager()
        self.dynamic_ai_manager = DynamicAIManager(self.model_manager)
    
    async def load_pipelines(self, pipeline_strings):
        self.dynamic_ai_manager.reload_active_config()
        self.dynamic_ai_manager.set_known_pipelines(pipeline_strings)
        constructed = self._construct_pipelines(pipeline_strings)
        if not constructed:
            raise ServerStopException("Error: No valid pipelines loaded!")

        self.model_manager.compute_vram_batch_sizes(self._collect_ai_processors([pipeline for _, pipeline in constructed]))
        started = await self._start_constructed_pipelines(constructed)
        if not started:
            raise ServerStopException("Error: No valid pipelines loaded!")
        self.pipelines = started

    async def reconfigure_pipelines(self, pipeline_strings):
        old_processors = self._collect_processors(self.pipelines.values())
        self.dynamic_ai_manager.reload_active_config()
        self.dynamic_ai_manager.set_known_pipelines(pipeline_strings)
        constructed = self._construct_pipelines(pipeline_strings)
        if not constructed:
            raise ServerStopException("Error: No valid pipelines loaded!")

        new_pipelines = [pipeline for _, pipeline in constructed]
        new_processors = self._collect_processors(new_pipelines)
        self.model_manager.compute_vram_batch_sizes(self._collect_ai_processors(new_pipelines))
        started = await self._start_constructed_pipelines(constructed)
        if not started:
            raise ServerStopException("Error: No valid pipelines loaded!")
        self.pipelines = started
        await self._stop_unused_processors(old_processors - new_processors)

    def _construct_pipelines(self, pipeline_strings):

        # Phase 1: Construct all pipelines (creates models, wires DAG).
        # No models are loaded to GPU yet, so we can count all models
        # before computing VRAM-aware batch sizes.
        constructed = []
        for pipeline in pipeline_strings:
            self.logger.info(f"Loading pipeline: {pipeline}")
            if not isinstance(pipeline, str):
                raise ValueError("Pipeline names must be strings that are the name of the pipeline config file!")
            pipeline_config_path = f"./config/pipelines/{pipeline}.yaml"
            try:
                loaded_config = load_config(pipeline_config_path)
                newpipeline = Pipeline(loaded_config, self.model_manager, self.dynamic_ai_manager, pipeline_name=pipeline)
                constructed.append((pipeline, newpipeline))
            except NoActiveModelsException as e:
                raise e
            except Exception as e:
                error_msg = str(e)
                if "No active AI models matched dynamic expansion filters" in error_msg:
                    self.logger.warning(
                        f"Pipeline '{pipeline}' skipped: no active models are available for one or more "
                        f"dynamic stages (models may not be downloaded or not listed in active_ai_models). "
                        f"Detail: {error_msg}"
                    )
                else:
                    self.logger.error(f"Error loading pipeline {pipeline}: {e}")
                    self.logger.debug("Exception details:", exc_info=True)

        return constructed

    async def _start_constructed_pipelines(self, constructed):
        started = {}
        for pipeline, newpipeline in constructed:
            try:
                await newpipeline.start_model_processing()
                started[pipeline] = newpipeline
                self.logger.info(f"Pipeline {pipeline} V{newpipeline.version} loaded successfully!")
            except NoActiveModelsException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error starting pipeline {pipeline}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
        return started

    def _collect_processors(self, pipelines):
        processors = set()
        for pipeline in pipelines:
            for model in getattr(pipeline, "models", []) or []:
                processor = getattr(model, "model", None)
                if processor is not None:
                    processors.add(processor)
        return processors

    def _collect_ai_processors(self, pipelines):
        return [processor for processor in self._collect_processors(pipelines) if getattr(processor, "is_ai_model", False)]

    async def _stop_unused_processors(self, processors):
        for processor in processors:
            try:
                await processor.stop_workers()
            except Exception as e:
                self.logger.error(f"Error stopping removed model processor: {e}")
                self.logger.debug("Exception details:", exc_info=True)

    def has_pipeline(self, pipeline_name) -> bool:
        return pipeline_name in self.pipelines

    async def stop_pipelines(self):
        pipelines = list(self.pipelines.items())
        for _, pipeline in pipelines:
            try:
                await pipeline.stop_model_processing()
            except Exception as e:
                self.logger.error(f"Error stopping pipeline {pipeline.short_name}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
        self.pipelines.clear()

    def get_pipeline(self, pipeline_name) -> Pipeline:
        if not pipeline_name in self.pipelines:
            self.logger.error(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
            raise ValueError(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
        pipeline = self.pipelines[pipeline_name]
        return pipeline

    async def get_request_future(self, data, pipeline_name):
        if not pipeline_name in self.pipelines:
            self.logger.error(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
            raise ValueError(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
        pipeline = self.pipelines[pipeline_name]
        futureData = {}
        if len(data) != len(pipeline.inputs):
            self.logger.error(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
            raise ValueError(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
        for inputName, inputData in zip(pipeline.inputs, data):
            futureData[inputName] = inputData
        futureData["pipeline"] = pipeline
        itemFuture = await ItemFuture.create(None, futureData, pipeline.event_handler)
        return itemFuture