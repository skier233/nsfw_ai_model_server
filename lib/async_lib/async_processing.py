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
        if self.data is None or self.future.done():
            return
        self.data[key] = value
        await self.handler(self, key)

    async def __setitem__(self, key, value):
        await self.set_data(key, value)

    def close_future(self, value):
        if self.future.done():
            return
        self.data = None
        self.future.set_result(value)

    def set_exception(self, exception):
        if self.future.done():
            return
        self.data = None
        self.future.set_exception(exception)
    
    def __getitem__(self, key):
        if self.data is None:
            return None
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
        self._loading_event = None
        self.is_ai_model = isinstance(self.model, AIModel)
        self.batch_collect_timeout = getattr(self.model, "batch_collect_timeout", 0.01)
        self.worker_tasks = []
        self.stopped = False

    def update_values_from_child_model(self, reset_queue=True):
        self.instance_count = self.model.instance_count
        if reset_queue:
            if self.model.max_queue_size is None:
                self.queue = asyncio.Queue()
            else:
                self.queue = asyncio.Queue(maxsize=self.model.max_queue_size)
        self.max_batch_size = self.model.max_batch_size
        self.max_batch_waits = self.model.max_batch_waits
        self.batch_collect_timeout = getattr(self.model, "batch_collect_timeout", 0.01)
        self.stopped = False
        
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
            skipped_categories = _resolve_optional_future_value(
                item.item_future,
                item.input_names,
                ["dynamic_skipped_categories", "skipped_categories"],
            )
            if skipped_categories is not None:
                this_ai_categories = self.model.model_category or []
                if this_ai_categories and all(this_category in skipped_categories for this_category in this_ai_categories):
                    await self.complete_item(item)
                    return True

            requested_model_names = _resolve_optional_future_value(
                item.item_future,
                item.input_names,
                ["dynamic_requested_model_names", "requested_model_names"],
            )
            if requested_model_names:
                normalized_requested = {str(name).strip() for name in requested_model_names if str(name).strip()}
                candidate_names = {
                    str(getattr(self.model, "config_name", "") or "").strip(),
                    str(getattr(self.model, "model_file_name", "") or "").strip(),
                }
                candidate_names.discard("")
                if candidate_names and candidate_names.isdisjoint(normalized_requested):
                    await self.complete_item(item)
                    return True
        batch_data.append(item)
        return False

    async def worker_process(self):
        while True:
            try:
                firstItem = await self.queue.get()
            except asyncio.CancelledError:
                break
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
                except asyncio.CancelledError:
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
            if self._loading_event is not None:
                await self._loading_event.wait()
            return
        else:
            try:
                self._loading_event = asyncio.Event()
                self.workers_started = True
                self.stopped = False
                await self.model.load()
                for _ in range(self.instance_count):
                    task = asyncio.create_task(self.worker_process())
                    self.worker_tasks.append(task)
                self._loading_event.set()
            except Exception as e:
                self.failed_loading = True
                if self._loading_event is not None:
                    self._loading_event.set()
                raise e

    async def stop_workers(self):
        if self.stopped:
            return

        self.stopped = True
        tasks = list(self.worker_tasks)
        self.worker_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        while True:
            try:
                pending_item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                try:
                    pending_item.item_future.set_exception(RuntimeError("Pipeline is reloading"))
                finally:
                    self.queue.task_done()

        self.workers_started = False
        self.failed_loading = False
        self._loading_event = None
        await self.model.unload()

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


def _resolve_optional_future_value(item_future, input_names, preferred_keys):
    for key in preferred_keys:
        value = item_future[key]
        if value is not None:
            return value

    for input_name in input_names:
        value = item_future[input_name]
        if value is None:
            continue
        for preferred_key in preferred_keys:
            if input_name == preferred_key or input_name.endswith(preferred_key):
                return value
    return None
