import asyncio
import csv
from typing import Any
import aiohttp
from pydantic import BaseModel

class VideoResult(BaseModel):
    result: Any

class ImageResult(BaseModel):
    result: Any

API_BASE_URL = 'http://localhost:8000'

async def call_api_async(session, endpoint, payload):
    url = f'{API_BASE_URL}/{endpoint}'
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {"error": f"Failed to process {endpoint}", "status_code": response.status}

async def process_images_async(image_paths):
    async with aiohttp.ClientSession() as session:
        return await call_api_async(session, 'process_images/', {"paths": image_paths, "threshold": 0.5, "return_confidence": True})

async def process_videos_async(video_paths):
    async with aiohttp.ClientSession() as session:
        return await call_api_async(session, 'process_video/', {"path": video_paths, "frame_interval": 2, "threshold": 0.5, "return_confidence": False})
    
def save_video_results_to_csv(video_results):
    # # Write results to CSV
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for result in video_results.result:
            row = [result['frame_index']] + result['actions']
            writer.writerow(row)


def main():
    image_paths = ['image_path1', 'image_path2', 'invalidpath']
    video_path = 'video_path1'

    image_results = ImageResult(**asyncio.run(process_images_async(image_paths)))
    video_results = VideoResult(**asyncio.run(process_videos_async(video_path)))
    save_video_results_to_csv(video_results)

    print("Image Processing Results:", image_results)
    print("Video Processing Results:", video_results)

if __name__ == "__main__":
    main()