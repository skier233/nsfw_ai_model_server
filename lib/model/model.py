import logging


class Model:
    def __init__(self, configValues):
        self.max_queue_size = configValues.get("max_queue_size", None)
        self.max_batch_size = configValues.get("max_batch_size", 1)
        self.instance_count = configValues.get("instance_count", 1)
        self.max_batch_waits = configValues.get("max_batch_waits", -1)
        self.logger = logging.getLogger("logger")

    async def worker_function_wrapper(self, data):
        try:
            await self.worker_function(data)
        except Exception as e:
            for item in data:
                item.item_future.set_exception(e)

    async def worker_function(self, data):
        pass

    async def load(self):
        return