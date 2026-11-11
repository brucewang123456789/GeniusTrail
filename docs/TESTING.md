# Source testing

Run these commands from the repository root on Linux x86-64.

```bash
python3 scripts/verify-source.py
bash scripts/bootstrap-linux.sh
bash scripts/test-source.sh
```

The source runner covers Voder's Node packages, including the CLI, provider bridge, SDK, agent core, search, memory, Modal transport fixtures and the source editor adapter. It also runs the Rust workspace, native gateway, and retrieval tests. Fixtures use local processes or mocked external services.

For the document and archive tests, install Poppler, LibreOffice, Tesseract, libarchive and Python with Pillow, xlrd and CairoSVG. The archive stress test extracts a 300 MiB file to a temporary directory. Ensure the host has room for the build and the temporary test data. Tests requiring root exercise additional ownership cases and report skips when run without root.

The source runner uses an isolated home for application settings. Live provider and Modal tests require operator credentials and run separately.
