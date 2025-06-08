import torch

# Load the checkpoint
checkpoint = torch.load('src/models/rgcn_linkpred.pt')

# Print the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# If the checkpoint is a state dict, print the keys and shapes
if isinstance(checkpoint, dict):
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

# If the checkpoint is a model state dict, look for conv1.convs keys
if isinstance(checkpoint, dict) and any(k.startswith('conv1.convs') for k in checkpoint.keys()):
    for key, value in checkpoint.items():
        if key.startswith('conv1.convs') and 'root' in key:
            print(f"{key}: {value.shape}") 