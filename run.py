#!/usr/bin/env python3
"""
Simple launcher for AI Video Reel Generator
Quick access to all features
"""

import os
import sys
from pathlib import Path

def main():
    """Simple launcher"""
    print("🎬 AI Video Reel Generator")
    print("="*40)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run from the project directory")
        sys.exit(1)
    
    print("🚀 Launching main application...")
    
    # Import and run main
    try:
        from main import main as main_app
        main_app()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()