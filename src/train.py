# ======================= Module Imports ============================= #
print("============ Importing modules... ============")
# ========== Base Modules
from datetime import datetime
import time

import numpy as np
from pathlib import Path
import os
import pandas as pd # for data handling

# ========== Modules for Data Analysis and handling
import re # for regex operations to analyze captions
from collections import Counter # to count word frequencies
from sklearn.model_selection import train_test_split # for splitting dataset
import itertools

# ========== Deep Learning Modules
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========== Models
from transformers import CLIPProcessor, CLIPModel

# ========== Custom Module for dataloader
import importlib
import dataset
importlib.reload(dataset)
from dataset import Flickr8kDataset
import utils
importlib.reload(utils)
from utils import *

# ========== Visualization Modules
from PIL import Image # for image processing
import matplotlib.pyplot as plt # for visualization

print("Modules imported.\n")



# =========================== Path Settings ========================== #
print("============ Setting up paths... ============ ")
# ============= set project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # fallback for Jupyter notebooks
    PROJECT_ROOT = find_project_root()

# ============= PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "flickr8k_captions.csv"
RUNS_DIR = PROJECT_ROOT / "runs"

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")
print(f"DATA_DIR set to: {DATA_DIR}")
print(f"IMG_DIR set to: {IMG_DIR}")
print(f"CAPTIONS_FILE set to: {CAPTIONS_FILE}")

print("\nPaths set.\n")

# ============================= Flags and presets =============================== #
print("============ Setting up flags... ============ ")

# ========= Run name
run_name = "test-run_v2"

# ========= Finetuning flag
finetuning = True

# ========= model saving flag
save_model = False
save_training_data = True

# ========= DataLoader and subset parameters
use_subset = True
subset_size = 100  # Number of samples in subset
batch_size = 32 # for DataLoader

# ========= Plotting parameters
save_plot = True
show_plot = True

# ======== Hyperparameter tuning
hyperparameter_tuning = True

# ======== Cluster settings
cluster = False
if cluster:
    num_workers = 4
else:
    num_workers = 0


print(f"Finetuning: {finetuning}")
print(f"Save Model: {save_model}")
print(f"Use Subset: {use_subset} ({subset_size})")
print(f"Batch Size: {batch_size}")
print(f"Hyperparameter Tuning: {hyperparameter_tuning}")
print(f"Cluster Settings: {cluster} (Num Workers: {num_workers})")
print("\nFlags set.\n")


# =============================== Load Dataset and prepare dataloader =============================== #
print("============ Loading dataset and preparing dataloaders... ============ ")
# =========== load csv file with captions
captions = pd.read_csv(CAPTIONS_FILE, sep=";", engine="python")

# =========== length of dataset
dataset_length = len(captions)



# ========== Subset option
# subset option
if use_subset:
    unique_images = captions['filename'].unique() # get unique image filenames
    chosen = pd.Series(unique_images).sample(n=subset_size, random_state=42) # randomly choose a subset
    subset = captions[captions['filename'].isin(chosen)].reset_index(drop=True)# filter captions to only include chosen images
    captions = subset.copy()  # update captions to only include subset




# ========= get unique images for proper splitting
images = captions["filename"].unique()

# ========= split into train, val, test
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

# ========= filter captions dataframe based on image splits
train_df = captions[captions["filename"].isin(train_imgs)]
val_df   = captions[captions["filename"].isin(val_imgs)]
test_df  = captions[captions["filename"].isin(test_imgs)]


# ========= create train, val and test datasets
train_dataset = Flickr8kDataset(str(IMG_DIR), train_df)
val_dataset   = Flickr8kDataset(str(IMG_DIR), val_df)
test_dataset  = Flickr8kDataset(str(IMG_DIR), test_df)

# ========= create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ========= print dataset info
print(captions["filename"].nunique(), "images in subset")  # number of unique images in subset
# print("Number of captions per image in subset:", captions.groupby("filename").size().describe())# number of captions per image in subset


print("\nDataset loaded and dataloaders prepared.\n")


# ======================= Model and Processor Setup ========================== #
print("============ Setting up model and processor... ============ ")

# model_name = "openai/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(model_name)
# processor = CLIPProcessor.from_pretrained(model_name)
#
# if finetuning:
#     # freez model parameters
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # onyl train projection layers
#     for param in model.visual_projection.parameters():
#         param.requires_grad = True
#     for param in model.text_projection.parameters():
#         param.requires_grad = True
# else:
#     # freeze vision encoder
#     for param in model.vision_model.parameters():
#         param.requires_grad = False
#
#     # freeze text encoder
#     for param in model.text_model.parameters():
#         param.requires_grad = False
#
# # optimizer, loss function and other parameters
# loss_fn = nn.CrossEntropyLoss()
epochs = 2
lr = 5e-5
batch_size = batch_size

# if finetuning:
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# else:
#     optimizer = optim.AdamW(model.text_model.parameters(), lr=lr)


# move model to device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(DEVICE)
# model.train()

# prepare filename for saving
filename = compile_filename(subset_size, DEVICE, batch_size, epochs, lr, run_name)

print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")
print(f"Batch Size: {batch_size}")
print("\nModel and processor set up.\n")
print(f"Run named as: {filename}")




# ============================ Training Loop ============================ #
print("============ Starting training... ============ ")
if hyperparameter_tuning:
    grid = {
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "batch_size": [16, 32, 64],
        "num_epochs": [2]
    }

    best_score = -1
    best_params = None
    best_model_state = None

    for lr, bs, ep in itertools.product(
            grid["learning_rate"],
            grid["batch_size"],
            grid["num_epochs"]
    ):
        print(f"\n--- Training with lr: {lr}, bs: {bs}, epochs: {ep} ---\n")
        logs, model = train_model(finetuning, train_dataset, val_dataset, ep, lr, bs, filename, DEVICE, RUNS_DIR, save_model, save_training_data, patience=5)


        score = logs["recall@1"].iloc[-1]

        if score > best_score:
            best_score = score
            best_params = {"lr": lr, "bs": bs, "ep": ep}
            best_model_state = model.state_dict()

    torch.save(best_model_state, RUNS_DIR / "best_model.pth")
    print(best_params, best_score)



else:
    logs, model = train_model(train_dataset, val_dataset, epochs, lr, batch_size, filename, DEVICE, RUNS_DIR, save_model, save_training_data, patience=5)

if save_training_data:
    model_file = RUNS_DIR / filename / f"{filename}_logs.csv"
    logs.to_csv(model_file, index=False)
    print(f"Training logs saved to {model_file}")


print("\nTraining completed.\n")

# ============================ Plotting Results ============================ #
print("============ Plotting training results... ============ ")
# plot training results
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(logs["epoch"], logs["train_loss"], label="Train Loss")
ax.plot(logs["epoch"], logs["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"Training and Validation Loss over Epochs - {run_name}")
ax.legend()
if save_plot:
    plot_file = RUNS_DIR / filename / f"{filename}_loss-plot.png"
    plt.savefig(plot_file)
    print(f"Loss plot saved to {plot_file}")
if show_plot:
    plt.show()


# plot recall metrics
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(logs["epoch"], logs["recall@1"], label="Recall@1")
ax.plot(logs["epoch"], logs["recall@5"], label="Recall@5")
ax.plot(logs["epoch"], logs["recall@10"], label="Recall@10")
ax.set_xlabel("Epoch")
ax.set_ylabel("Recall")
ax.set_title(f"Recall over Epochs - {run_name}")
ax.legend()
if save_plot:
    plot_file = RUNS_DIR / filename / f"{filename}_recall-plot.png"
    plt.savefig(plot_file)
    print(f"Recall plot saved to {plot_file}")
if show_plot:
    plt.show()
