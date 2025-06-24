# test/conftest.py
import os
# Set it at the module import stage to ensure that os.getenv can be obtained when any subsequent test module is imported
os.environ.setdefault("VELTRAX_API_TOKEN", "dummy_token")
