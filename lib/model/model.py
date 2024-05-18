class Model:
    def __init__(self, configValues):
        self.max_queue_size = configValues.get("max_queue_size", None)
        self.max_batch_size = configValues.get("max_batch_size", 1)
        self.instance_count = configValues.get("instance_count", 1)
        self.max_batch_waits = configValues.get("max_batch_waits", -1)

    async def worker_function(self, data):
        pass

    async def load(self):
        return