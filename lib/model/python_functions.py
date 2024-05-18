
import asyncio


async def result_coalescer(data):
    for item in data:
        itemFuture = item.item_future
        result = {}
        for input_name in item.input_names:
            result[input_name] = itemFuture[input_name]
        await itemFuture.set_data(item.output_names[0], result)
        
async def result_finisher(data):
    for item in data:
        itemFuture = item.item_future
        future_results = itemFuture[item.input_names[0]]
        itemFuture.close_future(future_results)

async def batch_awaiter(data):
    for item in data:
        itemFuture = item.item_future
        futures = itemFuture[item.input_names[0]]
        results = await asyncio.gather(*futures, return_exceptions=True)
        await itemFuture.set_data(item.output_names[0], results)

async def video_result_postprocessor(data):
    for item in data:
        itemFuture = item.item_future
        await itemFuture.set_data(item.output_names[0], itemFuture[item.input_names[0]])