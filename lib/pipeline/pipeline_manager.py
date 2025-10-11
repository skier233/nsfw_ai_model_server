
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
        for pipeline in pipeline_strings:
            self.logger.info(f"Loading pipeline: {pipeline}")
            if not isinstance(pipeline, str):
                raise ValueError("Pipeline names must be strings that are the name of the pipeline config file!")
            pipeline_config_path = f"./config/pipelines/{pipeline}.yaml"
            try:
                loaded_config = load_config(pipeline_config_path)
                newpipeline = Pipeline(loaded_config, self.model_manager, self.dynamic_ai_manager)
                self.pipelines[pipeline] = newpipeline
                await newpipeline.start_model_processing()
                self.logger.info(f"Pipeline {pipeline} V{newpipeline.version} loaded successfully!")
            except NoActiveModelsException as e:
                raise e
            except Exception as e:
                del self.pipelines[pipeline]
                self.logger.error(f"Error loading pipeline {pipeline}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
            
        if not self.pipelines:
            raise ServerStopException("Error: No valid pipelines loaded!")
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