import subprocess
import sys
from install import clean, build

def publish() -> None:
    '''
    Uploads the built package to PyPI using twine.
    Assumes TWINE_USERNAME and TWINE_PASSWORD are set in the environment or a .pypirc file is configured.
    '''
    print('Publishing package...')
    try:
        subprocess.run([sys.executable, '-m', 'twine', 'upload', 'dist/*'], check=True)
        print('Publish successful.')
    except subprocess.CalledProcessError as e:
        print(f'Publish failed with error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    clean()
    build()
    publish()
    clean()
