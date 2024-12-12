import sys
import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
sys.path.append("..")
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, model, layer, monitor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda import amp
import os
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import os
import shutil
import random

torch.manual_seed(3404)  # Set seed for generating random numbers for CPU and CUDA to ensure reproducibility
np.random.seed(3404)
torch.backends.cudnn.deterministic = True  # Use the same core allocation method for CUDA
torch.backends.cudnn.benchmark = False  # Do not optimize convolution and other operations at the hardware level
random.seed(3404)

# Run this after running the script:
# 1. Create a folder named "frames_number_10_split_by_number" in both generated folders.
# 2. Place the ten folders generated in the two folders into the created folder above.
# 3. Copy the "download", "events_np", and "extract" files into the two generated folders to prevent SpikingJelly from automatically downloading the dataset.

# Set the path of the raw data folder
data_folder = '../datasets/cifar10dvs/frames_number_10_split_by_number'

# Create train_dataset and test_dataset folders
train_dataset_folder = '../datasets/cifar10dvs/train_dataset'
test_dataset_folder = '../datasets/cifar10dvs/test_dataset'
os.makedirs(train_dataset_folder, exist_ok=True)
os.makedirs(test_dataset_folder, exist_ok=True)

# Get the list of folders in the raw data folder
data_folders = os.listdir(data_folder)

# Loop through each folder
for folder_name in data_folders:
    # Create corresponding folders in train_dataset and test_dataset
    train_folder = os.path.join(train_dataset_folder, folder_name)
    test_folder = os.path.join(test_dataset_folder, folder_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get the list of .npz files in the original folder
    npz_files = os.listdir(os.path.join(data_folder, folder_name))

    # Shuffle the list of files randomly
    random.shuffle(npz_files)

    # Calculate the split index
    split_index = int(0.9 * len(npz_files))

    # Split files into train_dataset and test_dataset folders
    for i, npz_file in enumerate(npz_files):
        source_file = os.path.join(data_folder, folder_name, npz_file)
        if i < split_index:
            target_file = os.path.join(train_folder, npz_file)
        else:
            target_file = os.path.join(test_folder, npz_file)
        shutil.copy(source_file, target_file)

print("Data splitting completed.")
