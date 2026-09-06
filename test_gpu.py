import torch
import time

from brats import get_datasets
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss


print("Starting...")

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("ERROR: CUDA is not available")
    exit()

# Dataset
print("Loading one patient...")

dataset = get_datasets(
    r"C:\Users\chand\BrainTumorProject\dataset",
    "train",
    target_size=(128, 128, 128)
)

sample = dataset[0]

image = sample["image"].unsqueeze(0).to(device)
label = sample["label"].unsqueeze(0).to(device)

print("Input:", image.shape)
print("Label:", label.shape)

# Model
model = SegResNet(
    spatial_dims=3,
    init_filters=32,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1)
).to(device)

print("Model loaded.")

# Loss
loss_func = DiceLoss(sigmoid=True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# Forward
print("Starting forward pass...")

start = time.time()

model.train()
logits = model(image)

torch.cuda.synchronize()

print("Forward completed in:", round(time.time() - start, 2), "seconds")
print("Output:", logits.shape)

# Loss
print("Starting loss...")

loss = loss_func(logits, label)

print("Loss:", loss.item())

# Backward
print("Starting backward pass...")

start = time.time()

loss.backward()

torch.cuda.synchronize()

print("Backward completed in:", round(time.time() - start, 2), "seconds")

# Optimizer
print("Starting optimizer step...")

optimizer.step()
optimizer.zero_grad()

print("Optimizer completed.")

print()
print("TEST SUCCESSFUL")