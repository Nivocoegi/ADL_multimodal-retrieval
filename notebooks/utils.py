from datetime import datetime
from pathlib import Path
import os


# ========= find project root
def find_project_root(start: Path = Path.cwd()):
    """Find the project root directory by looking for common markers."""
    # look for common project markers and return the first match
    for p in [start] + list(start.parents):
        if (p / 'data').exists() or (p / '.git').exists() or (p / 'pyproject.toml').exists():
            return p.resolve()
    return Path.cwd().resolve()



# ========== compile model filename
def compile_filename(device, batch, epochs, lr, description):
    today = datetime.now().strftime("%Y%m%d")
    file_name = f"{today}_clip_{device}_bs{batch}_ep{epochs}_lr{lr}_{description}"
    return file_name


if __name__ == "__main__":

    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.save_pretrained("clip-vit-base-patch32")
    processor.save_pretrained("clip-vit-base-patch32")

    from safetensors.torch import load_file
    import torch

    weights = load_file("/Users/nicolasvogel/Dokumente/16_ZHAW_MSc/V5_9_Adcanced_Deep_Learning/ADL_multimodal-retrieval_Misc/clip-vit-base-patch32/model.safetensors")
    torch.save(weights, "pytorch_model.bin")