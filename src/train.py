import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Flickr8kDataset
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import os
from pathlib import Path


def compile_filename(device, batch, epochs, lr, description):
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{today}_clip_{device}_bs{batch}_ep{epochs}_lr{lr}_{description}.pth"
    return file_name



def main():
    # --- 1. Konfiguration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-4

    # --- 2. Pfade dynamisch ---
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    IMG_DIR = DATA_DIR / "Images"
    CAPTIONS_FILE = DATA_DIR / "flickr8k_captions.csv"

    # --- 3. Model & Processor ---
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Encoder einfrieren (Variante B)
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = False

    # --- 4. Dataset & DataLoader ---
    dataset = Flickr8kDataset(root_dir=str(IMG_DIR), captions_file=str(CAPTIONS_FILE))

    print(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # num_workers=0 für macOS

    # --- 5. Optimizer & Loss ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.to(DEVICE)
    model.train()

    train_data = []

    # --- 6. Training Loop ---
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, texts in loop:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            ground_truth = torch.arange(len(images), device=DEVICE)
            loss_i = loss_fn(logits_per_image, ground_truth)
            loss_t = loss_fn(logits_per_text, ground_truth)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            train_data.append(total_loss)

        print(f"Epoch {epoch+1} — Avg Loss: {total_loss/len(loader):.4f}")

    filename = compile_filename(DEVICE, BATCH_SIZE, EPOCHS, LR, "test")

    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


if __name__ == '__main__':
    main()



learning_rates = (1e-5, 5e-6, 1e-6)
batch_sizes = (16, 32)
epochs = (5, 8, 10)