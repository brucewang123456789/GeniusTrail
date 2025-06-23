"""Package metadata for Veltraxor."""

from setuptools import setup, find_packages

setup(
    name="veltraxor",
    version="0.1.0",
    description="CLI / API chatbot with Dynamic CoT controller",
    author="YOUR_NAME",
    python_requires=">=3.9",
    packages=find_packages(where="."),
    py_modules=[
        "veltraxor",
        "llm_client",
        "dynamic_cot_controller",
        "api_server",
        "config",
    ],
    package_dir={"": "."},
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.25.1",
        "httpx>=0.27.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.21.0",
        "pydantic>=1.10.0",
        "pydantic-settings>=2.1.0",
        "jsonschema>=4.0.0",
        "redis>=4.5.0",
        "prometheus-client>=0.19.0",
        "langdetect>=1.0.9",
    ],
    extras_require={
        # Development dependencies, including HTTP mocking library
        "dev": [
            "pytest>=7.0",  # for running tests
            "pytest-asyncio>=0.23",  # for async tests
            "mypy>=1.5",  # for type checking
            "ruff>=0.0.241",  # for linting
            "black>=23.9.1",  # for code formatting
            "types-setuptools>=67.0.0",
            "types-requests>=2.25.0",
            "psycopg2-binary>=2.9.0",
            "wheel>=0.40",
            "build>=1.0",
            "respx>=0.22.0",  # HTTP mocking for Grok API tests
        ],
    },
    entry_points={
        "console_scripts": [
            "veltraxor=veltraxor:main",
        ],
    },
)
