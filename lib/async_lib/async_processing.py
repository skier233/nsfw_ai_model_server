import asyncio
import logging
import time

from lib.model.ai_model import AIModel
from lib.model.skip_input import Skip

logger = logging.getLogger("logger")

class ItemFuture:
    def __init__(self, parent, event_handler):
        self.parent = parent
        self.handler = event_handler
        self.future = asyncio.Future()
        self.data = {}
        root_candidate = self._resolve_root_future(parent)
        if root_candidate is None:
            root_candidate = self
        self.root_future = root_candidate
        if self.root_future is self:
            self._metrics_started_at = time.perf_counter()
    
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

    @staticmethod
    def _resolve_root_future(parent):
        if parent is None:
            return None
        if isinstance(parent, ItemFuture):
            return parent.root_future
        parent_future = getattr(parent, "item_future", None)
        if isinstance(parent_future, ItemFuture):
            return parent_future.root_future
        return None

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
        self.failed_loading = False
        self.is_ai_model = isinstance(self.model, AIModel)
        self.batch_collect_timeout = getattr(self.model, "batch_collect_timeout", 0.01)

    def update_values_from_child_model(self):
        self.instance_count = self.model.instance_count
        if self.model.max_queue_size is None:
            self.queue = asyncio.Queue()
        else:
            self.queue = asyncio.Queue(maxsize=self.model.max_queue_size)
        self.max_batch_size = self.model.max_batch_size
        self.max_batch_waits = self.model.max_batch_waits
        self.batch_collect_timeout = getattr(self.model, "batch_collect_timeout", 0.01)
        
    async def add_to_queue(self, data):
        await self.queue.put(data)

    async def add_items_to_queue(self, data):
        for item in data:
            await self.queue.put(item)

    async def complete_item(self, item):
        for output in item.output_names:
            await item.item_future.set_data(output, Skip())

    async def batch_data_append_with_skips(self, batch_data, item):
        if self.is_ai_model:
            skipped_categories = item.item_future[item.input_names[3]]
            if skipped_categories is not None:
                this_ai_categories = self.model.model_category
                if all(this_category in skipped_categories for this_category in this_ai_categories):
                    await self.complete_item(item)
                    return True
        batch_data.append(item)
        return False

    async def worker_process(self):
        while True:
            firstItem = await self.queue.get()
            batch_data = []
            if await self.batch_data_append_with_skips(batch_data, firstItem):
                self.queue.task_done()
                continue

            while len(batch_data) < self.max_batch_size:
                try:
                    if self.batch_collect_timeout <= 0:
                        next_item = self.queue.get_nowait()
                    else:
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=self.batch_collect_timeout)
                except asyncio.QueueEmpty:
                    break
                except asyncio.TimeoutError:
                    break

                if await self.batch_data_append_with_skips(batch_data, next_item):
                    self.queue.task_done()
                    continue
            
            if len(batch_data) > 0:
                start_time = time.perf_counter() if self.is_ai_model else None
                try:
                    await self.model.worker_function_wrapper(batch_data)
                finally:
                    if self.is_ai_model and start_time is not None:
                        elapsed = time.perf_counter() - start_time
                        self._record_ai_runtime(batch_data, elapsed)
                    for _ in batch_data:
                        self.queue.task_done()

    async def start_workers(self):
        if self.workers_started:
            if self.failed_loading:
                raise Exception("Error: Model failed to load!")
            return
        else:
            try:
                self.workers_started = True
                await self.model.load()
                for _ in range(self.instance_count):
                    asyncio.create_task(self.worker_process())
                    self.workers_started = True
            except Exception as e:
                self.failed_loading = True
                raise e

    def _record_ai_runtime(self, batch_data, elapsed):
        if elapsed <= 0:
            return
        root_counts = {}
        for item in batch_data:
            root_future = getattr(item.item_future, "root_future", None) or item.item_future
            root_counts[root_future] = root_counts.get(root_future, 0) + 1

        total_items = sum(root_counts.values())
        if total_items == 0:
            return

        for root_future, count in root_counts.items():
            metrics = getattr(root_future, "_pipeline_metrics", None)
            if metrics is None:
                metrics = {}
                setattr(root_future, "_pipeline_metrics", metrics)
            metrics["ai_inference_seconds"] = metrics.get("ai_inference_seconds", 0.0) + (elapsed * (count / total_items))
