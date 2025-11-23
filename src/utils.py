# Utility functions for training and evaluating models
import time
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from torch import nn, optim
from torch.utils.data import DataLoader



# ================== Main Functions ======================== #

# =========== Training function with early stopping and metric logging
def train_model(finetuning, train_dataset, val_dataset, num_epochs, lr, batch_size, filename, DEVICE, RUNS_DIR, save_model, save_training_data, patience=5):

    # setting up model, processor and optimizer
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(DEVICE)
    model.train()

    if finetuning:
        # freez model parameters
        for param in model.parameters():
            param.requires_grad = False

        # onyl train projection layers
        for param in model.visual_projection.parameters():
            param.requires_grad = True
        for param in model.text_projection.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        # freeze vision encoder
        for param in model.vision_model.parameters():
            param.requires_grad = False

        # freeze text encoder
        for param in model.text_model.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(model.text_model.parameters(), lr=lr)

    # optimizer, loss function and other parameters
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    logs = []

    # ---------- METRIC FUNCTION ----------
    def compute_metrics(model, processor, dataloader):
        model.eval()
        all_image_embs = []
        all_text_embs = []

        with torch.no_grad():
            for images, texts in dataloader:
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

                img = model.get_image_features(pixel_values=inputs["pixel_values"])
                txt = model.get_text_features(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])

                img = img / img.norm(dim=-1, keepdim=True)
                txt = txt / txt.norm(dim=-1, keepdim=True)

                all_image_embs.append(img)
                all_text_embs.append(txt)

        all_image_embs = torch.cat(all_image_embs)
        all_text_embs = torch.cat(all_text_embs)

        sims = (all_image_embs * all_text_embs).sum(dim=-1)
        mean_sim = sims.mean().item()

        recall_at_k = {}
        sim_matrix = all_image_embs @ all_text_embs.T

        for k in [1, 5, 10]:
            topk = sim_matrix.topk(k, dim=1).indices
            correct = sum(i in topk[i] for i in range(len(all_image_embs)))
            recall_at_k[f"recall@{k}"] = correct / len(all_image_embs)

        return mean_sim, recall_at_k

    # ------------------ TRAINING LOOP -------------------
    for epoch in range(num_epochs):
        start = time.time()

        # ----- TRAIN -----
        model.train()
        train_loss = 0

        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)

            logits_i = outputs.logits_per_image
            logits_t = outputs.logits_per_text

            gt = torch.arange(len(images), device=DEVICE)
            loss = (criterion(logits_i, gt) + criterion(logits_t, gt)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, texts in val_loader:
                inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)

                logits_i = outputs.logits_per_image
                logits_t = outputs.logits_per_text

                gt = torch.arange(len(images), device=DEVICE)
                loss = (criterion(logits_i, gt) + criterion(logits_t, gt)) / 2

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # -------- METRICS AFTER EACH EPOCH --------
        mean_sim, rec = compute_metrics(model, processor, val_loader)

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_similarity": mean_sim,
            "recall@1": rec["recall@1"],
            "recall@5": rec["recall@5"],
            "recall@10": rec["recall@10"],
            "time_sec": time.time() - start
        })

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MeanSim: {mean_sim:.4f} | "
            f"R@1: {rec['recall@1']:.3f} | "
            f"Epoch time: {time.time() - start:.2f}s"
        )

        # -------- EARLY STOPPING --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # restore best model
    model.load_state_dict(best_model_state)

    save_path = RUNS_DIR / filename
    save_path.mkdir(parents=True, exist_ok=True)
    logs = pd.DataFrame(logs)
    # save model
    if save_model:
        model_file = save_path / f"{filename}_model.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Model saved to {model_file}")

    if save_training_data:
        model_file = RUNS_DIR / filename / f"{filename}_logs.csv"
        logs.to_csv(model_file, index=False)
        print(f"Training logs saved to {model_file}")

    return logs, model


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
    today = datetime.now().strftime("%Y%m%d")
    file_name = f"{today}_clip_{device}_sub-size{subset_size}_bs{batch}_ep{epochs}_lr{lr}_{description}"
    return file_name
