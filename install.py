"""
Orchify Install Script
Usage: python install.py
"""

import sys
import subprocess
import os


def install_package():
    """Install orchify package."""
    print("Installing orchify...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", script_dir],
            cwd=script_dir,
        )
        print("Successfully installed orchify!")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)


def verify_installation():
    """Verify that orchify can be imported."""
    print("\nVerifying installation...")
    
    try:
        import orchify
        print(f"orchify {orchify.__version__} installed successfully")
        print(f"Available exports: {', '.join(orchify.__all__)}")
        return True
    except ImportError as e:
        print(f"Failed to import orchify: {e}")
        return False


def main():
    install_package()
    
    if verify_installation():
        print("\nInstallation complete! You can now use:")
        print("  from orchify import Agent, Tool, Group")
    else:
        print("\nWarning: Package installed but verification failed.")
        print("You may need to check your Python environment.")


if __name__ == "__main__":
    main()
