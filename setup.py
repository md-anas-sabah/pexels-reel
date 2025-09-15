#!/usr/bin/env python3
"""
Setup script for AI Video Reel Generator
Installs dependencies and prepares the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"📍 Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python dependencies")
        return False

def check_ffmpeg():
    """Check FFmpeg installation"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("📝 Install instructions:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["output_reels", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")

def check_env_file():
    """Check .env file"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for required keys
        with open(env_file) as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OpenAI API key configured")
            else:
                print("⚠️  OpenAI API key not found in .env")
                print("💡 Add: OPENAI_API_KEY='your-key-here'")
    else:
        print("⚠️  .env file not found")
        print("💡 Create .env file with:")
        print("   OPENAI_API_KEY='your-openai-key'")

def create_launch_scripts():
    """Create convenient launch scripts"""
    print("🚀 Creating launch scripts...")
    
    # Windows batch file
    batch_content = """@echo off
echo 🎬 AI Video Reel Generator
python main.py
pause
"""
    
    with open("launch.bat", "w") as f:
        f.write(batch_content)
    print("✅ Created: launch.bat (Windows)")
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "🎬 AI Video Reel Generator"
python3 main.py
"""
    
    with open("launch.sh", "w") as f:
        f.write(shell_content)
    
    # Make executable
    try:
        os.chmod("launch.sh", 0o755)
        print("✅ Created: launch.sh (macOS/Linux)")
    except:
        print("⚠️  Created launch.sh but couldn't make executable")

def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        import requests
        from crewai import Agent
        print("✅ All imports successful")
        
        # Test OpenCV cascades
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if Path(face_cascade_path).exists():
            print("✅ OpenCV cascades available")
        else:
            print("⚠️  OpenCV cascades not found")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🛠️  AI Video Reel Generator Setup")
    print("="*50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg missing - install manually")
    
    # Create directories
    create_directories()
    
    # Check environment
    check_env_file()
    
    # Create launch scripts
    create_launch_scripts()
    
    # Run tests
    if success:
        if not run_tests():
            success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Make sure FFmpeg is installed")
        print("2. Add your OpenAI API key to .env file")
        print("3. Run: python main.py")
        print("\n🚀 Ready to create amazing reels!")
    else:
        print("❌ Setup completed with errors")
        print("💡 Please fix the issues above and try again")
    
    print("="*50)

if __name__ == "__main__":
    main()