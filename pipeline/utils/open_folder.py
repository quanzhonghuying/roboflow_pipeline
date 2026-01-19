import os
import subprocess
import sys


def open_folder(path: str) -> None:
    """Open a folder in system file explorer."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["explorer", path], check=False)
    else:
        # Linux
        subprocess.run(["xdg-open", path], check=False)
