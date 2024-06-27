import torch

#Two helper functions created to help train a model locally

def get_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory

        print(f"Total GPU memory: {total_memory / 1e9} GB")
        print(f"Reserved GPU memory: {reserved_memory / 1e9} GB")
        print(f"Allocated GPU memory: {allocated_memory / 1e9} GB")
        print(f"Free GPU memory: {free_memory / 1e9} GB")
    else:
        print("No GPU available.")

def estimate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    
    # Assuming float32 (4 bytes) for parameters and buffers
    model_size = (total_params + total_buffers) * 4
    print(f"Estimated Model Size: {model_size / (1024 ** 2):.2f} MB, total number of parameters: {model_size:,}")
    return model_size