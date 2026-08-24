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
    except: pass

def build() -> None:
    '''
    Runs the setup.py sdist bdist_wheel command to build the package.
    '''
    print('Building package...')
    try:
        subprocess.run([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'], check=True)
        print('Build successful.')
    except subprocess.CalledProcessError as e:
        print(f'Build failed with error: {e}')
        sys.exit(1)

def install() -> None:
    '''
    Installs the package from the built distribution.
    '''
    print('Installing package...')
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation'], check=True)
        print('Installation successful.')
    except subprocess.CalledProcessError as e:
        print(f'Installation failed with error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    install()
    clean()
