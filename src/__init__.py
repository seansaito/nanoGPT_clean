from pathlib import Path

# Path to the root directory (the repo)
ROOT_DIR = Path(__file__).parent.parent

# Other directories
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = DATA_DIR / "log"
CONFIG_DIR = ROOT_DIR / "config"

# Encoding
ENCODING = "utf-8-sig"
