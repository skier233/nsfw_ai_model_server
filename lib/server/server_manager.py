import asyncio
from contextlib import asynccontextmanager
import gc
import logging
import os
import signal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from lib.config.config_utils import load_config
from lib.logging.logger import setup_logger
from lib.pipeline.pipeline_manager import PipelineManager
from lib.server.exceptions import ServerStopException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import requests

def get_latest_release_version(logger):
    api_url = "https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    try:
        response = requests.get(api_url)
        latest_release = response.json()
        return latest_release['tag_name']
    except:
        logger.warning("Failed to get the latest release version from GitHub to see if an update is needed. Are you offline?")

class ServerManager:
    def __init__(self):
        config_path = "./config/config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path, default_config={})
        else:
            ServerStopException(f"Main config file does not exist: {config_path}")
        loglevel = config.get("loglevel", "INFO")
        setup_logger("logger", loglevel)
        self.logger = logging.getLogger("logger")
        version = config.get("VERSION", "1.3.1")
        latest_version = get_latest_release_version(self.logger)
        self.logger.debug(f"Current version: {version}, Latest version: {latest_version}")
        if version != latest_version:
            self.logger.warning("There is a new version available! Please update the server using update.sh or update.ps1")
        self.config = config
        self.pipeline_manager = PipelineManager()
        self.default_image_pipeline = config.get("default_image_pipeline", None)
        if self.default_image_pipeline is None:
            self.logger.error("No default image pipeline found in the configuration file.")
            raise ServerStopException("No default image pipeline found in the configuration file.")
        
        self.default_video_pipeline = config.get("default_video_pipeline", None)
        if self.default_video_pipeline is None:
            self.logger.error("No default video pipeline found in the configuration file.")
            raise ServerStopException("No default video pipeline found in the configuration file.")

    async def startup(self):
        pipelines = self.config["active_pipelines"]
        if not pipelines:
            self.logger.error("No pipelines found in the configuration file.")
            raise ServerStopException("No pipelines found in the configuration file.")
        await self.pipeline_manager.load_pipelines(pipelines)
        self.logger.info("Pipelines loaded successfully")
        self.background_task = asyncio.create_task(check_inactivity())

    async def get_request_future(self, data, pipeline_name):
        return await self.pipeline_manager.get_request_future(data, pipeline_name)
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    await server_manager.startup()
    yield
    pass

class OutstandingRequestsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.outstanding_requests = 0
        self.last_request_timestamp = asyncio.get_event_loop().time()

    async def dispatch(self, request: Request, call_next):
        self.outstanding_requests += 1
        self.last_request_timestamp = asyncio.get_event_loop().time()
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse({"detail": str(e)}, status_code=500)
        self.outstanding_requests -= 1
        return response
    
app = FastAPI(lifespan=lifespan)

outstanding_requests_middleware = OutstandingRequestsMiddleware(app)
app.add_middleware(BaseHTTPMiddleware, dispatch=outstanding_requests_middleware.dispatch)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    response.headers["X-Outstanding-Requests"] = str(outstanding_requests_middleware.outstanding_requests)
    return response

last_request_timestamp = 0

async def check_inactivity():
    global last_request_timestamp
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        middleware = outstanding_requests_middleware
        if middleware.outstanding_requests == 0 and asyncio.get_event_loop().time() - middleware.last_request_timestamp > 300 and middleware.last_request_timestamp > last_request_timestamp:
            last_request_timestamp = middleware.last_request_timestamp
            # no requests in the last 10 minutes
            print("No requests in the last 5 minutes, Clearing cached memory")
            torch.cuda.empty_cache()
            gc.collect()

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