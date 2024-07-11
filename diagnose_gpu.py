import torch

print (torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())   
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
print(f"Allocated memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
print(f"Cached memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
print(f"Free memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3):.2f} GB")