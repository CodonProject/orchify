import glob
import os
import shutil
import subprocess
import sys

def clean() -> None:
    '''
    Cleans up the build, dist, and egg-info directories.
    '''
    dirs_to_remove = ['build', 'dist']
    for root, dirs, _ in os.walk('.'):
        for name in dirs:
            if name.endswith('.egg-info'):
                dirs_to_remove.append(os.path.join(root, name))

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            print(f'Removing {dir_path}...')
            shutil.rmtree(dir_path)
    try:
        subprocess.run(['codon', 'clear'], check=True)
    except (OSError, subprocess.SubprocessError):
        pass

def build() -> None:
    '''
    Builds the sdist and the wheel with the standard PEP 517 frontend.

    Raises:
        SystemExit: When the build frontend is missing or the build fails.
    '''
    print('Building package with python -m build...')
    try:
        subprocess.run([sys.executable, '-m', 'build'], check=True)
        print('Build successful.')
    except FileNotFoundError:
        print('The build frontend is missing; run: pip install build')
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f'Build failed with error: {e}')
        sys.exit(1)

def install(wheel: bool = True) -> None:
    '''
    Installs the newest built wheel, or the local source tree.

    Args:
        wheel (bool): Prefer the newest ``dist/*.whl`` over the source tree.

    Raises:
        SystemExit: When the installation fails.
    '''
    wheels = sorted(glob.glob(os.path.join('dist', '*.whl')))
    target = wheels[-1] if wheel and wheels else '.'
    print(f'Installing {target}...')
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--force-reinstall', target],
            check=True,
        )
        print('Installation successful.')
    except subprocess.CalledProcessError as e:
        print(f'Installation failed with error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    build()
    install()
    clean()
