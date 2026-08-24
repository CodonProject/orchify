import os
import re
from typing import List
from setuptools import setup, find_packages

def get_version() -> str:
    '''
    Reads the version from orchify/__init__.py.

    Returns:
        str: The version string.
    '''
    init_path = os.path.join('orchify', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError('Unable to find version string.')

def get_requirements() -> List[str]:
    '''
    Reads the requirements from requirement.txt.

    Returns:
        List[str]: A list of package dependencies.
    '''
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='orchify',
    version=get_version(),
    description='LLM package',
    author='Yizhou Li',
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.8'
)
