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


from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained("clip-vit-base-patch32")