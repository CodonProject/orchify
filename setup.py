'''
Compatibility shim.

All metadata now lives in pyproject.toml; this file only keeps
`python setup.py sdist bdist_wheel` (used by install.py/publish.py) working.
Prefer:
    python -m build
    python -m pip install .
'''
from setuptools import setup

setup()
