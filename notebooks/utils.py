from datetime import datetime
from pathlib import Path
import os


def find_project_root(start: Path = Path.cwd()):
    """Find the project root directory by looking for common markers."""
    # look for common project markers and return the first match
    for p in [start] + list(start.parents):
        if (p / 'data').exists() or (p / '.git').exists() or (p / 'pyproject.toml').exists():
            return p.resolve()
    return Path.cwd().resolve()



def compile_filename(device, batch, epochs, lr, description):
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{today}_clip_{device}_bs{batch}_ep{epochs}_lr{lr}_{description}.pth"
    return file_name

