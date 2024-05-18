import asyncio
import csv
import time
import requests
import aiohttp

async def call_process_images_api_async(session, image_paths):
    url = 'http://localhost:8000/process_images/'
    payload = {
        "paths": image_paths
    }
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {"error": "Failed to process images", "status_code": response.status}
        
async def call_process_videos_api_async(session, video_paths):
    url = 'http://localhost:8000/process_video/'
    payload = {
        "path": video_paths
    }
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {"error": "Failed to process videos", "status_code": response.status}
        
async def batching_test(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [call_process_images_api_async(session, image_paths) for _ in range(12)]
        results = await asyncio.gather(*tasks)
        return results


# Example usage
image_paths = ['image_path1', 'image_path2', 'invalidpath']

#video_paths = 'video_path1'
curr = time.time()
results = asyncio.run(batching_test(image_paths))
newTime = time.time()
print(results)
print(f"Processed {len(image_paths)} in: ", newTime - curr)
#video_results = call_process_videos_api(video_paths)
# newTime = time.time()
# print("Processed video in: ", newTime - curr)

# print("Video Processing Results:", video_results)

# #video_results = call_process_videos_api(video_paths)
# #print("Video Processing Results:", video_results)

# # Write results to CSV
# with open('results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)


#     for result in video_results:
#         row = [result['frame_index'] / 30] + result['actions']
#         writer.writerow(row)