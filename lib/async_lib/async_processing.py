import asyncio

class ItemFuture:
    def __init__(self, parent, event_handler):
        self.parent = parent
        self.handler = event_handler
        self.future = asyncio.Future()
        self.data = {}
    
    async def set_data(self, key, value):
        self.data[key] = value
        await self.handler(self, key)

    async def __setitem__(self, key, value):
        await self.set_data(key, value)

    def close_future(self, value):
        self.data = None
        self.future.set_result(value)

    def set_exception(self, exception):
        self.data = None
        self.future.set_exception(exception)
    
    def __getitem__(self, key):
        return self.data.get(key)

    def __await__(self):
        yield from self.future.__await__()

        return self.future.result()

    @classmethod
    async def create(cls, parent, data, event_handler):
        self = ItemFuture(parent, event_handler)
        for key in data:
            await self.set_data(key, data[key])
        return self

class QueueItem:
    def __init__(self, itemFuture, input_names, output_names):
        self.item_future = itemFuture
        self.input_names = input_names
        self.output_names = output_names

class ModelProcessor():
    def __init__(self, model):
        self.model = model
        self.instance_count = model.instance_count
        if model.max_queue_size is None:
            self.queue = asyncio.Queue()
        else:
            self.queue = asyncio.Queue(maxsize=model.max_queue_size)
        self.max_batch_size = self.model.max_batch_size
        self.max_batch_waits = self.model.max_batch_waits
        self.workers_started = False

    async def add_to_queue(self, data):
        await self.queue.put(data)

    async def add_items_to_queue(self, data):
        for item in data:
            await self.queue.put(item)

    async def worker_process(self):
        while True:
            firstItem = await self.queue.get()
            batch_data = []
            batch_data.append(firstItem)

            waitsSoFar = 0

            while len(batch_data) < self.max_batch_size:
                if not self.queue.empty():
                    batch_data.append(await self.queue.get())
                elif waitsSoFar < self.max_batch_waits:
                    waitsSoFar += 1
                    await asyncio.sleep(1)
                else:
                    break

            await self.model.worker_function(batch_data)

            for _ in batch_data:
                self.queue.task_done()

    async def start_workers(self):
        if self.workers_started:
            return
        else:
            self.workers_started = True
        await self.model.load()
        for _ in range(self.instance_count):
            asyncio.create_task(self.worker_process())
            self.workers_started = True