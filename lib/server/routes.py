import asyncio
from lib.server.api_definitions import ImagePathList, VideoPathList
from lib.server.server_manager import server_manager, app

@app.post("/process_images/")
async def process_images(request: ImagePathList):
    image_paths = request.paths
    futures = [await server_manager.get_request_future([path], "image_pipeline_giddy") for path in image_paths]
    results = await asyncio.gather(*futures, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            results[i] = {"error": str(result)}

    return results

@app.post("/process_video/")
async def process_video(request: VideoPathList):
    data = [request.path]
    future = await server_manager.get_request_future(data, "video_pipeline_giddy")
    result = await future
    return result