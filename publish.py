import glob
import os
import subprocess
import sys

from install import build, clean

def publish() -> None:
    '''
    Uploads every file in dist/ to PyPI with twine.

    Expects TWINE_USERNAME and TWINE_PASSWORD in the environment, or a
    configured .pypirc file.

    Raises:
        SystemExit: When dist/ is empty or the upload fails.
    '''
    packages = sorted(glob.glob(os.path.join('dist', '*')))
    if not packages:
        print('Nothing to publish in dist/; run the build first.')
        sys.exit(1)
    print(f'Publishing {len(packages)} file(s)...')
    try:
        subprocess.run(
            [sys.executable, '-m', 'twine', 'upload', *packages, '--non-interactive'],
            check=True,
        )
        print('Publish successful.')
    except subprocess.CalledProcessError as e:
        print(f'Publish failed with error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    clean()
    build()
    publish()
    clean()
