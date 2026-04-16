
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
        self.dynamic_ai_manager.set_known_pipelines(pipeline_strings)

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

        if not constructed:
            raise ServerStopException("Error: No valid pipelines loaded!")

        # Phase 2: Compute optimal batch sizes using VRAM budget.
        # Now that all models are registered, we can estimate total weight
        # memory and allocate batch sizes that account for all loaded models.
        self.model_manager.compute_vram_batch_sizes()

        # Phase 3: Start all pipelines (loads models to GPU, starts workers).
        for pipeline, newpipeline in constructed:
            try:
                await newpipeline.start_model_processing()
                self.pipelines[pipeline] = newpipeline
                self.logger.info(f"Pipeline {pipeline} V{newpipeline.version} loaded successfully!")
            except NoActiveModelsException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error starting pipeline {pipeline}: {e}")
                self.logger.debug("Exception details:", exc_info=True)

        if not self.pipelines:
            raise ServerStopException("Error: No valid pipelines loaded!")

    def has_pipeline(self, pipeline_name) -> bool:
        return pipeline_name in self.pipelines

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