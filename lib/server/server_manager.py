from contextlib import asynccontextmanager
import os
import signal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from lib.config.config_utils import load_config
from lib.pipeline.pipeline_manager import PipelineManager
from lib.server.exceptions import ServerStopException
from fastapi.middleware.cors import CORSMiddleware

class ServerManager:
    def __init__(self):
        self.pipeline_manager = PipelineManager()

    async def startup(self):
        config_path = "./config/config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path, default_config={})
        else:
            ServerStopException(f"Main config file does not exist: {config_path}")
        pipelines = config["active_pipelines"]
        if not pipelines:
            print("No pipelines found in the configuration file.")
            raise ServerStopException("No pipelines found in the configuration file.")
        await self.pipeline_manager.load_pipelines(pipelines)

    async def get_request_future(self, data, pipeline_name):
        return await self.pipeline_manager.get_request_future(data, pipeline_name)
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    await server_manager.startup()
    yield
    pass
app = FastAPI(lifespan=lifespan)
server_manager = ServerManager()

origins = [
    "*",  # Replace with the actual origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including OPTIONS
    allow_headers=["*"],
)

@app.exception_handler(ServerStopException)
async def server_stop_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"Server is stopping due to error: {exc.message}"},
    )