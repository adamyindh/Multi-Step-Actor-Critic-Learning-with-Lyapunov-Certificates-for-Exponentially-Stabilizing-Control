"""
Purpose of the init.py file: Mark a directory as a Python package.
This allows Python modules (.py files) in the directory to be imported.
Without this file, Python treats the directory as a regular folder rather than an importable package.
Even an empty init.py file tells Python: "This directory should be treated as a Python package".
After installation via pip install -e ., all subpackages will be correctly recognized.
"""