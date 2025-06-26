import sys
from pathlib import Path
import importlib

# Add the project root directory to sys.path so that the api_server module can be imported
proj_root = Path(__file__).parent.parent
proj_root_str = str(proj_root.resolve())
if proj_root_str not in sys.path:
    sys.path.insert(0, proj_root_str)

# Dynamically import the api_server module in the root directory and extract the FastAPI instance
_main = importlib.import_module("api_server")
app = getattr(_main, "app")
