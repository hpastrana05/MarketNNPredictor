import torch

# 1. Basic check
print(f"Is CUDA available? {torch.cuda.is_available()}")

# 2. Get the name of the GPU
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 3. Simple Tensor Test
    # Create a tensor and move it to the GPU
    x = torch.tensor([1.0, 2.0]).to("cuda")
    print(f"Tensor is on device: {x.device}")
else:
    print("CUDA is NOT reaching your GPU.")