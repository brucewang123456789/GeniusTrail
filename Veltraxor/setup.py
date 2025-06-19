from setuptools import setup

# NOTE:
#  - We use py_modules for single-file modules at repo root.
#  - If you later move to a proper package directory (src layout),
#    change to packages=find_packages(...).

setup(
    name="veltraxor",
    version="0.1.0",
    description="CLI / API chatbot with Dynamic CoT controller",
    author="YOUR_NAME",
    python_requires=">=3.9",
    py_modules=[
        "veltraxor",
        "llm_client",
        "dynamic_cot_controller",
        "api_server",
        "config",
    ],
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.25.1",
        "httpx>=0.27.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "pydantic-settings>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "wheel>=0.40",
            "build>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # veltraxor --help
            "veltraxor=veltraxor:main",
        ],
    },
)