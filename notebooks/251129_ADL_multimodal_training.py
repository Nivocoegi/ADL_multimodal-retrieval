#!/usr/bin/env python
# coding: utf-8

# # Advacned Deep Learning - Multimodal Retrieval with CLIP on Flickr8k
# 
# Author: Nicolas Vogel
# 
# Submission date: 15.12.2025
# 
# course: Advanced Deep Learning (AS 2025)
# 
# ## Sections
# 0. Introduction
# 1. Pre-Setting, Imports and Data Loading
# 2. Data Analysis and Visualization
# 3. Data Preprocessing
# 4. Model Building and Training
# 5. Evaluation (in separate file)
# 6. Final Remarks
# 
# # 0. Introduction
# ## 0.1 Taks
# The objective of this project is to develop and evaluate a multimodal retrieval system capable of matching images and text descriptions. The system is built using a pretrained CLIP model and fine-tuned on the Flickr8k dataset to improve cross-modal alignment between visual and textual representations.
# 
# The task includes preparing and analyzing the dataset, implementing preprocessing pipelines, and adapting the CLIP architecture for efficient fine-tuning. A custom training loop is used to optimize the model for the retrieval task, including loss computation for image–text matching and various training strategies such as freezing, unfreezing, and progressive unfreezing of model components.
# 
# After training, the system is evaluated through retrieval experiments where text queries are used to retrieve relevant images, and images are used to retrieve matching captions. Performance is assessed using standard retrieval metrics such as Recall@K and mean similarity, allowing quantitative comparison of different training configurations and hyperparameters.
# 
# The final system aims to demonstrate how effectively a pretrained multimodal model can be adapted to a smaller dataset and how design choices in training influence retrieval performance.
# 
# 
# ## 0.2 Related Work
# ### CLIP
# CLIP was created in 2021 by Radford and his team from OpenAI. CLIP learns visual models from natural language by training an image encoder and a text encoder together to predict the correct image and caption for each other. For training, they used a huge set of data from the internet – about 400 million image-text pairs. The model makes a shared space for both types. The authors show that the model can describe new visual concepts using natural language in a zero-shot manner. This means the encoder creates a linear classifier, which then predicts captions for images it hasn't seen before [1].
# 
# ### CLIP-branches
# One of the most important things CLIP can do, made possible by the multimodal embedding space, is finding text and images. Traditional image retrieval systems usually use something called a "nearest neighbour search algorithm" to find images based on the features they have. CLIP-based systems use this shared embedding space to let users search for images using text queries, or vice versa. However, the results of a traditional nearest neighbours search are often not very precise or relevant, which can lead to answers that are not very good. These answers might include objects that are not relevant or include examples that are not similar. Lülf and his team looked at this issue and tried to improve how accurate and complete CLIP-based retrieval systems are.
# The developed CLIP-Branches is a new search engine that searches for text and images. It works by getting people to interact with the search engine to give feedback on how relevant the results are. This approach lets users improve their search by giving them positive and negative examples, which leads to better results that look at all the data, not just the first results [2].
# 
# 
# ### Flickr8k Dataset
# The dataset contains 8,000 images collected from Flickr, each paired with five different captions. The images cover a wide range of scenes and objects, especially outdoor activities, people, and animals as later can be seen in the analysis of the dataset. The captions are human-generated and provide descriptive information about the content of the images [3].
# 
# 
# 
# 
# 
# [1] A. Radford et al., „Learning Transferable Visual Models From Natural Language Supervision“, 26. Februar 2021, arXiv: arXiv:2103.00020. doi: 10.48550/arXiv.2103.00020.
# 
# [2] C. Lülf, D. M. L. Martins, M. A. V. Salles, Y. Zhou, und F. Gieseke, „CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval“, 19. Juni 2024, arXiv: arXiv:2406.13322. doi: 10.48550/arXiv.2406.13322.
# 
# [3] https://github.com/Avaneesh40585/Flickr8k-Dataset
# 
# 
# ## 0.3 Sumbission Data
# 
# 
# 
# 
# 

# # 1     Pre-Setting, Imports and Data Loading
# In this section the necessary modules are imported, paths are set up, and the Flickr8k dataset is loaded for further processing. Also some helper functions are defined to facilitate model saving and project structure management.
# 
# ## 1.1 Modules and helper functions

# In[13]:


# ======================= Module Imports ============================= #

# ========== Base Modules
from datetime import datetime # for naming files with date
import time # for measuring training time
import math
import numpy as np
from pathlib import Path
import os
import pandas as pd # for data handling

# ========== Modules for Data Analysis and handling
import re # for regex operations to analyze captions
from collections import Counter # to count word frequencies
from sklearn.model_selection import train_test_split # for splitting dataset
import itertools # for hyperparameter tuning combinations

# ========== Deep Learning Modules
import torch #
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm # for progress bars

# ========== Models
from transformers import CLIPProcessor, CLIPModel # for CLIP model and processor


# ========== Custom Module for dataloader
import importlib # for reloading custom modules
import my_dataset
importlib.reload(my_dataset)
from my_dataset import Flickr8kDataset # Custom Dataset class
import utils
importlib.reload(utils)
from utils import *

# ========== Visualization Modules
from PIL import Image # for image processing
import matplotlib.pyplot as plt # for visualization


# In[14]:


# ================== Helperfunctions ======================== #

# ========= find project root
def find_project_root(start: Path = Path.cwd()):
    """Find the project root directory by looking for common markers."""
    # look for common project markers and return the first match
    for p in [start] + list(start.parents):
        if (p / 'data').exists() or (p / '.git').exists() or (p / 'pyproject.toml').exists():
            return p.resolve()
    return Path.cwd().resolve()

# ========== compile model filename
def compile_filename(subset_size, device, batch, epochs, lr, description):
    """Compile a filename for saving the model based on training parameters."""
    today = datetime.now().strftime("%Y%m%d")
    file_name = f"{today}_clip_{device}_sub-size{subset_size}_bs{batch}_ep{epochs}_lr{lr}_{description}"
    return file_name



# ## 1.2 Funcitonality Pre-settings
# Here paths and flags are set for data loading, model saving, and training configurations.

# In[15]:


# ===================== Path Settings ========================= #

# ============= set project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # fallback for Jupyter notebooks
    PROJECT_ROOT = find_project_root()

# ============= set data paths
DATA_DIR = PROJECT_ROOT / "data_2"
# DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "flickr30k_captions.csv"
# CAPTIONS_FILE = DATA_DIR / "flickr8k_captions.csv"
RUNS_DIR = PROJECT_ROOT / "runs"


# In[16]:


# ==================== Flags ======================= #

# ========= Finetuning flag
finetuning = True

# ========= model saving flag
save_model = True

# ========= DataLoader and subset parameters
use_subset = False
subset_size = "-full"  # Number of samples in subset
batch_size = 64 # for DataLoader
num_workers = 0  # for DataLoader # 0 for local; 4 for cluster

# ========= Plotting parameters
save_plot = True
show_plot = True

# ======== Hyperparameter tuning
hyperparameter_tuning = False


# ======== Cluster settings
cluster = True
if cluster:
    num_workers = 4

# ======== Train full model flag
train_full_model = False


# ========= Progressive unfreezing flag
progressive_unfreeze = True


# ========= Run name for saving
run_name = "flickr30k_prog_unfreeze"

# ## 1.3 Data Loading
# 
# Here the data is loaded from the Flickr8k dataset for further analysis and model training.

# In[17]:


# ================== Load Dataset ====================== #
# =========== load csv file with captions
captions = pd.read_csv(CAPTIONS_FILE, sep=";", engine="python")

# =========== length of dataset
dataset_length = len(captions)
print(f"Dataset contains {dataset_length} image-caption pairs.")

# =========== show first 5 entries
captions.head()


# # 2 Data Analysis and Visualization
# In this section the Flickr8k dataset is analyzed and visualized to gain insights into its structure and content. Various statistics about the images and captions are computed and visualized using plots.
# 
# ## 2.1 Examples

# In[18]:


# ========== Visualize Examples ====================== #
# ========== plot some random images with captions as example
fig, axes = plt.subplots(3, 3, figsize=(20, 5))

# random seed
# np.random.seed(42)

for ax in axes.ravel():
    row = captions.sample(1).iloc[0]
    img = Image.open(os.path.join(IMG_DIR, row['filename']))
    ax.imshow(img)
    ax.set_title(row['caption'])
    ax.axis('off')

plt.tight_layout()
plt.show()


# ## 2.2 Image and Caption analysis

# In[19]:


# ============ Check number of Captions per Image ============= #

# ========== plot histogram of number of captions per image
captions_per_image = captions.groupby("filename").size() # count captions per image
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(captions_per_image, bins=range(1, captions_per_image.max()+2), align='left', edgecolor='black') # plot histogram
ax.set_xlabel("Number of Captions per Image")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Captions per Image")
plt.xticks(range(1, captions_per_image.max()+1))
plt.show()



# As we can see, each image in the Flickr8k dataset has exactly 5 captions, which is consistent with the dataset's design.

# In[20]:


# ============ Word Frequency Analysis in Captions ============= #

# ======= Perform word frequency analysis
text = " ".join(captions["caption"].astype(str)).lower() # combine all captions into one large text
words = re.findall(r"\b\w+\b", text) # combine all captions into one large text

# ========== count word frequencies and save top 60 and least 60
freq = Counter(words)
top60 = freq.most_common(60) # get top 60 most common words
least60 = freq.most_common()[:-61:-1] # get least 60 common words

# ======= Plot the results
fig, axes = plt.subplots(2, 1, figsize=(12, 5))

# top: Top 30
axes[0].bar([w for w,_ in top60], [c for _,c in top60])
axes[0].set_title("Top 60")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis='x', rotation=90)

# bottom: Least 30
axes[1].bar([w for w,_ in least60], [c for _,c in least60])
axes[1].set_title("Least 60")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


# 

# In[21]:


# =========== Word Category Analysis in Top 60 Words ============= #
# closer look at nouns, verbs, adjectives in top 60 words
# manually categorized based on common English usage

# ====== top 60 words categorized
top60_nouns = [('dog', 8138), ('man', 7274), ('boy', 3581), ('woman', 3402), ('girl', 3328), ('people', 2883), ('water', 2790), ('dogs', 2125), ('shirt', 1962), ('ball', 1783), ('grass', 1622), ('snow', 1547), ('child', 1545), ('person', 1542), ('field', 1283), ('group', 1218), ('children', 1156)] # nouns

top60_verbs = [('is', 9345), ('are', 3504), ('wearing', 3062), ('running', 2073), ('playing', 2008), ('standing', 1787), ('jumping', 1473), ('sitting', 1368), ('holding', 1324), ('walking', 1165)] # verbs

top60_adj = [('white', 3959), ('black', 3848), ('red', 2691), ('brown', 2578), ('blue', 2279), ('little', 1768), ('small', 1278), ('large', 1236), ('green', 1234), ('yellow', 1217)] # adjectives

# ====== Plot the results
fig, axes = plt.subplots(3, 1, figsize=(12, 5))

# top: Top 30
axes[0].bar([w for w,_ in top60_nouns], [c for _,c in top60_nouns])
axes[0].set_title("Nouns out of top 60")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis='x', rotation=90)

# bottom: Least 30
axes[1].bar([w for w,_ in top60_verbs], [c for _,c in top60_verbs])
axes[1].set_title("Verbs out of top 60")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis='x', rotation=90)

# bottom: Least 30
axes[2].bar([w for w,_ in top60_adj], [c for _,c in top60_adj])
axes[2].set_title("adjectives out of top 60")
axes[2].set_ylabel("Frequency")
axes[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


# The most common words in the captions are primarily function words such as "the", "is", "a", and "in", which are essential for sentence structure but do not carry significant semantic content. Content words like "dog", "man", and "woman" also appear frequently, reflecting common subjects in the images. The least common words tend to be more specific nouns and adjectives that describe particular objects or attributes, indicating a long tail of less frequently mentioned concepts in the dataset.

# In[23]:


# ======= Generall Dataset Statistics ======= #

# ===== number of unique words
num_unique_words = len(freq)
print(f"Number of unique words in captions: {num_unique_words}")

# ===== average caption length
avg_caption_length = sum(len(caption.split()) for caption in captions["caption"]) / dataset_length
print(f"Average caption length: {avg_caption_length:.2f} words")

# ===== plot caption length distribution
caption_lengths = [len(caption.split()) for caption in captions["caption"]]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(caption_lengths, bins=range(1, max(caption_lengths)+2), align='left', edgecolor='black')
ax.set_xlabel("Caption Length (words)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Caption Lengths")
plt.xticks(range(1, max(caption_lengths)+1))
plt.show()



# Most of the captions in the Flickr8k dataset are relatively short, with an average length of around 12 words. The distribution of caption lengths shows that while many captions are concise, there is a significant number of longer captions that provide more detailed descriptions of the images. This variability in caption length reflects the diverse ways in which different annotators describe the same image, ranging from brief phrases to more elaborate sentences.

# In[24]:


# ======= Image Dimension Analysis ======= #

# ===== image proerties
image_sizes = [] # list to store image sizes
for filename in captions["filename"].unique(): # loop over unique images
    img_path = os.path.join(IMG_DIR, filename) # # get image path
    with Image.open(img_path) as img: # open image
        image_sizes.append(img.size) # append (width, height)
widths, heights = zip(*image_sizes) # unzip widths and heights
avg_width = sum(widths) / len(widths) # calculate average width
avg_height = sum(heights) / len(heights) # calculate average height
print(f"Average image size: {avg_height:.2f} x {avg_width:.2f} pixels") # print average size

# ===== plot image dimension distribution
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].hist(heights, bins=30,  edgecolor='black')
ax[0].axvline(avg_height, linestyle="--", color = 'red')
ax[0].set_xlabel("height (px)")
ax[0].set_ylabel("frequency")
ax[0].set_title("Distribution of Image Heights")
ax[0].legend(["average height"])

ax[1].hist(widths, bins=30,  edgecolor='black')
ax[1].axvline(avg_width, linestyle="--", color = 'red')
ax[1].set_xlabel("width (px)")
ax[1].set_ylabel("frequency")
ax[1].set_title("Distribution of Image Widths")
ax[1].legend(["average width"])

plt.tight_layout()
plt.show()


# The images in the Flickr8k dataset exhibit a wide range of dimensions, with heights and widths varying significantly across the dataset. The average image size is approximately 400 pixels in height and 450 pixels in width, indicating that while some images are relatively small, others are much larger. This variability in image size reflects the diverse nature of the images collected from Flickr, which include different scenes, objects, and compositions.
# Most of the images do have widths around 500 pixels, the hight is more variable 330 and 500 pixels seem to be common sizes.

# # 3 Data Preprocessing
# 
# ## 3.1 Subset
# For quicker experimentation, a subset of the dataset can be used. Here we randomly select a specified number of unique images and their corresponding captions to create a smaller dataset for training and evaluation.
# To check the subset some statistics are printed out.

# In[13]:


# ================= Create Subset ====================== #
# ======= subset option
if use_subset:
    unique_images = captions['filename'].unique() # get unique image filenames
    chosen = pd.Series(unique_images).sample(n=subset_size, random_state=42) # randomly choose a subset
    subset = captions[captions['filename'].isin(chosen)].reset_index(drop=True)# filter captions to only include chosen images
    captions = subset.copy()  # update captions to only include subset

# ======= print dataset info
print(captions["filename"].nunique(), "images in dataset")  # number of unique images in subset
print(captions.groupby("filename").size().describe())# number of captions per image in subset
# captions.head() # display first few entries of subset


# ## 3.2 Dataloader
# Here the dataset is split into training, validation, and test sets. Custom datasets and dataloaders are created for each split to facilitate model training and evaluation.

# In[14]:


# ================= Data Splitting and Dataloaders ====================== #

# ====== get unique images for proper splitting
images = captions["filename"].unique()

# ====== split into train, val, test
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

# ====== filter captions dataframe based on image splits
train_df = captions[captions["filename"].isin(train_imgs)]
val_df   = captions[captions["filename"].isin(val_imgs)]
test_df  = captions[captions["filename"].isin(test_imgs)]

# ====== create datasets
train_dataset = Flickr8kDataset(str(IMG_DIR), train_df)
val_dataset   = Flickr8kDataset(str(IMG_DIR), val_df)
test_dataset  = Flickr8kDataset(str(IMG_DIR), test_df)

# ====== create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



# # 4 Model Building and Training
# 
# ## 4.1 Loading Pretrained Model
# The pretrained CLIP model and processor are loaded, and the model is configured for fine-tuning or training specific layers based on the defined settings. The optimizer, loss function, and other training parameters are also set up.

# In[17]:


# ================= Load Pretrained CLIP Model ====================== #

# ===== Progressive Unfreezing Settings
unfreeze_every = 15                 # all x epoch more is unfreezed
unfreeze_stages = ["text", "vision", "all"]
current_stage = 0

# ===== model intilization
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
processor.tokenizer.model_max_length = 77 # set max token length
processor.tokenizer.truncation_side = "right" # set truncation side
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Temperature
temp = 0.12
model.logit_scale.data = torch.tensor(math.log(1 / temp)).to(DEVICE) # set initial temperature
model.logit_scale.requires_grad = False  # optional: freeze temperature parameter during training (No)


# ===== set finetuning or partial training
if not finetuning:
    # fine-tuning entire model
    for param in model.parameters():
        param.requires_grad = True
else:
    # freez model parameters
    for param in model.parameters():
        param.requires_grad = False

    # onyl train projection layers
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    for param in model.text_projection.parameters():
        param.requires_grad = True

# ====== optimizer, loss function and other parameters
loss_fn = nn.CrossEntropyLoss()
epochs = 30
lr = 5e-5
batch_size = batch_size

if finetuning:
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
else:
    optimizer = optim.AdamW(model.text_model.parameters(), lr=lr)

# ====== move model to device
model.to(DEVICE)
model.train()

# ====== prepare filename for saving
run_name = run_name
filename = compile_filename(subset_size, DEVICE, batch_size, epochs, lr, run_name)
print(f"Model will be saved as: {filename}")



# 
# ## 4.2 Training Loop

# In[16]:


# ================ Training Loop function ====================== #

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, filename, current_stage, unfreeze_every, unfreeze_stages, patience=5):

    # ======== Early Stopping and Logging Setup
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    logs = []

    # ======== FUNCTION TO COMPUTE METRICS
    def compute_metrics(model, processor, dataloader):

        # initialize
        model.eval()
        all_image_embs = []
        all_text_embs = []

        # compute embeddings
        with torch.no_grad():
            for images, texts in dataloader:
                inputs = processor(text=texts, images=images, padding=True, trunction=True, return_tensors="pt", padding=True).to(DEVICE)

                img = model.get_image_features(pixel_values=inputs["pixel_values"]) # get image features
                txt = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]) # get text features

                img = img / img.norm(dim=-1, keepdim=True) # normalize img
                txt = txt / txt.norm(dim=-1, keepdim=True) # normalize txt

                all_image_embs.append(img)
                all_text_embs.append(txt)

        # concatenate all embeddings
        all_image_embs = torch.cat(all_image_embs)
        all_text_embs = torch.cat(all_text_embs)

        # compute mean similarity
        sims = (all_image_embs * all_text_embs).sum(dim=-1)
        mean_sim = sims.mean().item()

        # compute Recall@K
        recall_at_k = {}
        sim_matrix = all_image_embs @ all_text_embs.T

        # Recall@K calculation for k = 1, 5, 10 to evaluate retrieval performance and plot results
        for k in [1, 5, 10]:
            topk = sim_matrix.topk(k, dim=1).indices
            correct = sum(i in topk[i] for i in range(len(all_image_embs)))
            recall_at_k[f"recall@{k}"] = correct / len(all_image_embs)

        return mean_sim, recall_at_k

    # =============== TRAINING LOOP
    for epoch in range(num_epochs):

        warmup_epochs = 30
        if progressive_unfreeze and epoch >= warmup_epochs:

            if (epoch - warmup_epochs) % unfreeze_every == 0 and current_stage < len(unfreeze_stages):

                stage = unfreeze_stages[current_stage]
                print(f"\n>>> Unfreezing stage: {stage}")

                if stage == "text":
                    for p in model.text_model.parameters():
                        p.requires_grad = True

                elif stage == "vision":
                    for p in model.vision_model.parameters():
                        p.requires_grad = True

                elif stage == "all":
                    for p in model.parameters():
                        p.requires_grad = True
                    model.logit_scale.requires_grad = False

                current_stage += 1

                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


        # ======== PROGRESSIVE UNFREEZING
        if progressive_unfreeze and epoch > 0:

            # unfreeze next stage
            if epoch % unfreeze_every == 0 and current_stage < len(unfreeze_stages):

                # unfreeze stage
                stage = unfreeze_stages[current_stage]
                print(f"\n>>> Unfreezing stage: {stage}")

                # text unfreezing
                if stage == "text":
                    for p in model.text_model.parameters():
                        p.requires_grad = True

                # vision unfreezing
                elif stage == "vision":
                    for p in model.vision_model.parameters():
                        p.requires_grad = True

                # all unfreezing
                elif stage == "all":
                    for p in model.parameters():
                        p.requires_grad = True
                    model.logit_scale.requires_grad = False  # optional: freez temperature

                current_stage += 1

                # refresh optimizer, so that new parameters get learned
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # ====== Epoch time measurement
        start = time.time()

        # ====== TRAIN
        model.train()
        train_loss = 0

        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False): # loop over batches of training data
            inputs = processor(images=images, text=texts, padding=True, trunction=True, return_tensors="pt").to(DEVICE) # prepare inputs
            outputs = model(**inputs) # forward pass

            logits_i = outputs.logits_per_image # image-to-text logits
            logits_t = outputs.logits_per_text # text-to-image logits

            gt = torch.arange(len(images), device=DEVICE) # ground truth labels
            loss = (criterion(logits_i, gt) + criterion(logits_t, gt)) / 2 # compute loss

            optimizer.zero_grad()
            loss.backward() # backward pass
            optimizer.step() # optimizer step

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======= VALIDATION
        model.eval()
        val_loss = 0

        # no grad for validation
        with torch.no_grad():
            for images, texts in val_loader: # loop over batches of validation data
                inputs = processor(images=images, text=texts, padding=True, trunction=True, return_tensors="pt").to(DEVICE) # prepare inputs
                outputs = model(**inputs) # forward pass

                logits_i = outputs.logits_per_image # image-to-text logits
                logits_t = outputs.logits_per_text # text-to-image logits

                gt = torch.arange(len(images), device=DEVICE) # ground truth labels
                loss = (criterion(logits_i, gt) + criterion(logits_t, gt)) / 2 # compute loss

                val_loss += loss.item() # accumulate validation loss

        val_loss /= len(val_loader) # average validation loss over batches

        # ======== METRICS AFTER EACH EPOCH
        mean_sim, rec = compute_metrics(model, processor, val_loader) # compute metrics

        # ==== LOGGING
        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_similarity": mean_sim,
            "recall@1": rec["recall@1"],
            "recall@5": rec["recall@5"],
            "recall@10": rec["recall@10"],
            "unfreeze_stage": stage if progressive_unfreeze else "none",
            "time_sec": time.time() - start
        })

        # ==== PRINT EPOCH STATS
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MeanSim: {mean_sim:.4f} | "
            f"R@1: {rec['recall@1']:.3f} | "
            f"Epoch time: {time.time() - start:.2f}s"
        )

        # ====== EARLY STOPPING
        if val_loss < best_val_loss: # check for improvement of validation loss
            best_val_loss = val_loss # update best validation loss
            best_model_state = model.state_dict() # save best model state
            patience_counter = 0 # reset patience counter
        else:
            patience_counter += 1 # increment patience counter
            if patience_counter >= patience: # check if patience exceeded
                print("Early stopping triggered.")
                break

    # restore best model
    model.load_state_dict(best_model_state)

    # ======== save model
    if save_model:
        save_path = RUNS_DIR / filename # define save path
        save_path.mkdir(parents=True, exist_ok=True) # create directory if not exists
        model_file = save_path / f"{filename}_model.pth" # define model file path
        torch.save(model.state_dict(), model_file) # save model state
        print(f"Model saved to {model_file}")

    return pd.DataFrame(logs)


# In[49]:


# ==================== Model Training ====================== #
#train model
if not hyperparameter_tuning:
   logs = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, filename, current_stage, unfreeze_every, unfreeze_stages, patience=50)


# ## 4.3 Saving the Model and export Data

# In[34]:


# ==================== Save Logs and Plots ====================== #
# save logs to csv
if not hyperparameter_tuning:
    if save_model:
        model_file = RUNS_DIR / filename / f"{filename}_logs.csv"
        logs.to_csv(model_file, index=False)
        print(f"Training logs saved to {model_file}")


# In[83]:


# plot training results
if not hyperparameter_tuning:
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
    if show_plot:
        plt.show()


# In[84]:


# plot recalls
if not hyperparameter_tuning:
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
    if show_plot:
        plt.show()


# 
# ## 4.3 Hyperparameter Tuning
# 

# In[ ]:


if hyperparameter_tuning:
    best_model = None
    best_performance = 0
    best_hyperparams = {}

    # Grid
    hyperparameter_grid = {
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "batch_size": [16, 32, 64],
        "num_epochs": [200]
    }
    # Number of Hyperparameter combinations
    total_runs = (len(hyperparameter_grid["learning_rate"]) *
          len(hyperparameter_grid["batch_size"]) *
          len(hyperparameter_grid["num_epochs"])
    )
    current_run = 1
    print(f"Total Hyperparameter combinations to try: {total_runs}")



    for lr, batch_size, epochs in itertools.product(
        hyperparameter_grid["learning_rate"],
        hyperparameter_grid["batch_size"],
        hyperparameter_grid["num_epochs"]
    ):
        today = datetime.now().strftime("%Y%m%d")
        run_name = f"{today}_HT-full_lr{lr}_bs{batch_size}_ep{epochs}"
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Training {run_name} | run {current_run}/{total_runs} ===")

        # ----- Reinitialize model + optimizer -----
        # if cluster:
        #     model_path = "/cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/models/clip-vit-base-patch32"
        #     model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        #     processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        # else:
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        if not finetuning:
            # fine-tuning entire model
            for param in model.parameters():
                param.requires_grad = True
        else:
            # freez parameters
            for param in model.parameters():
                param.requires_grad = False

            # only train projection layers
            for param in model.visual_projection.parameters():
                param.requires_grad = True
            for param in model.text_projection.parameters():
                param.requires_grad = True


        # Optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # data loader with new batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model.to(DEVICE)

        # ----- Train -----
        logs = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs,filename=run_name, patience=10)

        # ----- Save logs -----
        logs_file = run_dir / f"{run_name}_logs.csv"
        logs.to_csv(logs_file, index=False)
        print(f"Logs saved: {run_name}_logs.csv")

        # ----- Save model -----
        model_file = run_dir / f"{run_name}_model.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Model saved: {run_name}_model.pth")

        # ----- LOSS Plot -----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logs["epoch"], logs["train_loss"], label="Train Loss")
        ax.plot(logs["epoch"], logs["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss - {run_name}")
        ax.legend()
        plt.savefig(run_dir / f"{run_name}_loss_plot.png")
        plt.close()
        print(f"Loss-Plot saved")

        # ----- RECALL Plot -----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logs["epoch"], logs["recall@1"],  label="Recall@1")
        ax.plot(logs["epoch"], logs["recall@5"],  label="Recall@5")
        ax.plot(logs["epoch"], logs["recall@10"], label="Recall@10")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.set_title(f"Recalls - {run_name}")
        ax.legend()
        plt.savefig(run_dir / f"{run_name}_recall_plot.png")
        plt.close()
        print(f"Recall-Plot saved")

        # ----- Evaluate performance -----
        final_recall = logs["recall@1"].iloc[-1]

        if final_recall > best_performance:
            best_performance = final_recall
            best_model = model.state_dict()
            best_hyperparams = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": epochs
            }
        current_run += 1

    # ---------- SUMMARY ----------
    print("\n====================")
    print("   BEST RUN")
    print("====================")
    print(best_hyperparams)
    print(f"Best Recall@1: {best_performance:.4f}")


# # 5 Evaluation

# # 6 Final Remarks
