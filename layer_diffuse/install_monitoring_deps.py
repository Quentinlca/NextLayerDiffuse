#!/usr/bin/env python3
"""
Installation script for performance monitoring dependencies.
"""

import subprocess
import sys

def install_package(package_name, description=""):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        if description:
            print(f"   {description}")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully!")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False

def main():
    print("🚀 Setting up performance monitoring dependencies")
    print("=" * 60)
    
    packages = [
        ("nvidia-ml-py3", "NVIDIA GPU monitoring library"),
        ("psutil", "System and process monitoring"),
    ]
    
    success_count = 0
    for package, description in packages:
        if install_package(package, description):
            success_count += 1
        print()
    
    print(f"✨ Installation complete! {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("\n🎉 All dependencies installed successfully!")
        print("   You can now run: python performance_monitor_fixed.py")
    else:
        print("\n⚠️  Some packages failed to install.")
        print("   You can try installing them manually:")
        for package, _ in packages:
            print(f"   pip install {package}")
    
    print("\n📝 Alternative installation methods:")
    print("   For conda users: conda install nvidia-ml-py3 psutil")
    print("   For pip users: pip install nvidia-ml-py3 psutil")

if __name__ == "__main__":
    main()
