#!/usr/bin/env python3
"""
Main Entry Point for AI Video Reel Generator
Complete management system for video reel creation
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking system dependencies...")
    
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
        else:
            print("❌ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        return False
    
    # Check Python packages - map package names to import names
    required_packages = {
        "crewai": "crewai",
        "opencv-python": "cv2", 
        "numpy": "numpy",
        "requests": "requests",
        "pydantic": "pydantic",
        "python-dotenv": "dotenv"
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Add to .env file: OPENAI_API_KEY='your-key-here'")
        return False
    else:
        print("✅ OpenAI API key found")
    
    print("✅ All dependencies are ready!")
    return True

def setup_environment():
    """Setup the environment and create necessary directories"""
    print("🛠️  Setting up environment...")
    
    # Create output directory
    output_dir = Path("output_reels")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def display_main_menu():
    """Display the main menu"""
    print("\n" + "="*80)
    print("🎬 AI VIDEO REEL GENERATOR - MAIN MENU")
    print("="*80)
    print("📱 Transform any video into perfect Instagram Reels, TikTok & YouTube Shorts")
    print("🎯 Smart object detection • 720x1280 output • Professional quality")
    print("="*80)
    
    print("\n🚀 AVAILABLE MODES:")
    print("1. 📋 Interactive Mode - Full customization with categories & preferences")
    print("2. ⚡ Quick Mode - Fast conversion with simple search")
    print("3. 🎮 Demo Mode - See features and examples")
    print("4. 🔧 Settings - Check system status and configuration")
    print("5. 📊 View Results - See previously generated reels")
    print("6. ❓ Help - Usage guide and tips")
    print("0. 🚪 Exit")

def interactive_mode():
    """Run the interactive reel generator"""
    print("\n🎨 Starting Interactive Mode...")
    try:
        from interactive_reel_generator import ReelGeneratorUI
        generator = ReelGeneratorUI()
        generator.run()
    except ImportError as e:
        print(f"❌ Error importing interactive generator: {e}")
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")

def quick_mode():
    """Run quick mode with simple search"""
    print("\n⚡ Quick Mode - Fast Video Conversion")
    print("-" * 40)
    
    # Get search query
    query = input("🔍 Enter search query (e.g., 'sunset beach', 'city night'): ").strip()
    if not query:
        query = "nature landscape"
        print(f"📝 Using default query: '{query}'")
    
    # Get number of videos
    try:
        num_videos = int(input("📹 How many videos to convert? (1-5, default 2): ") or "2")
        num_videos = max(1, min(5, num_videos))  # Limit between 1-5
    except ValueError:
        num_videos = 2
        print(f"📝 Using default: {num_videos} videos")
    
    print(f"\n🚀 Searching for '{query}' and converting {num_videos} video(s)...")
    
    try:
        from video_reel_converter import VideoReelConverter
        
        # Initialize converter
        pexels_api_key = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
        converter = VideoReelConverter(pexels_api_key)
        
        # Convert videos
        results = converter.convert_to_reel(query, per_page=num_videos)
        
        if results:
            print(f"\n🎉 Success! Converted {len(results)} video(s)")
            display_results_summary(results)
        else:
            print("❌ No videos were converted. Try a different search query.")
            
    except Exception as e:
        print(f"❌ Error in quick mode: {e}")
    finally:
        try:
            converter.cleanup()
        except:
            pass

def demo_mode():
    """Show demo and features"""
    print("\n🎮 Demo Mode - Features Overview")
    try:
        from demo_interactive import show_features, run_demo
        run_demo()
    except ImportError:
        print("❌ Demo module not found")
    except Exception as e:
        print(f"❌ Error in demo mode: {e}")

def settings_mode():
    """Show settings and system status"""
    print("\n🔧 System Settings & Status")
    print("="*50)
    
    # System info
    print(f"📂 Working Directory: {os.getcwd()}")
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # Check dependencies
    check_dependencies()
    
    # API Keys status
    print("\n🔑 API Keys Status:")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API Key: ...{openai_key[-8:] if len(openai_key) > 8 else '***'}")
    else:
        print("❌ OpenAI API Key: Not configured")
    
    pexels_key = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
    print(f"✅ Pexels API Key: ...{pexels_key[-8:]}")
    
    # Directory status
    print("\n📁 Directories:")
    output_dir = Path("output_reels")
    if output_dir.exists():
        video_files = list(output_dir.glob("*.mp4"))
        print(f"✅ Output folder: {len(video_files)} video(s) created")
    else:
        print("📁 Output folder: Not created yet")
    
    # Configuration options
    print("\n⚙️  Configuration Options:")
    print("1. 🔄 Reinstall dependencies")
    print("2. 🧹 Clean output folder")
    print("3. 📋 Export configuration")
    print("0. 🔙 Back to main menu")
    
    choice = input("\nSelect option (0-3): ").strip()
    
    if choice == "1":
        reinstall_dependencies()
    elif choice == "2":
        clean_output_folder()
    elif choice == "3":
        export_configuration()

def view_results():
    """View previously generated reels"""
    print("\n📊 Previously Generated Reels")
    print("="*40)
    
    output_dir = Path("output_reels")
    if not output_dir.exists():
        print("📁 No output folder found. Generate some reels first!")
        return
    
    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        print("📹 No videos found. Create some reels first!")
        return
    
    print(f"Found {len(video_files)} generated reel(s):\n")
    
    for i, video_file in enumerate(sorted(video_files), 1):
        # Get file info
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        
        # Get video info using ffprobe if available
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
                width = video_stream['width']
                height = video_stream['height']
                
                print(f"{i}. 📹 {video_file.name}")
                print(f"   📐 Resolution: {width}x{height}")
                print(f"   ⏱️  Duration: {duration:.1f}s")
                print(f"   💾 Size: {file_size:.1f} MB")
                print(f"   📁 Path: {video_file}")
                print()
            else:
                print(f"{i}. 📹 {video_file.name} ({file_size:.1f} MB)")
        except:
            print(f"{i}. 📹 {video_file.name} ({file_size:.1f} MB)")
    
    print("💡 Tips:")
    print("  • These videos are ready for Instagram Reels, TikTok, YouTube Shorts")
    print("  • Always credit the original photographers when posting")
    print("  • Videos are optimized for mobile viewing")

def help_mode():
    """Show help and usage guide"""
    print("\n❓ Help & Usage Guide")
    print("="*40)
    
    print("\n🚀 QUICK START:")
    print("1. Choose 'Interactive Mode' for full customization")
    print("2. Select category (Nature, Urban, People, etc.)")
    print("3. Pick mood and style preferences")
    print("4. Preview and select videos")
    print("5. Get perfect 720x1280 reels!")
    
    print("\n📱 OUTPUT FORMAT:")
    print("• Resolution: 720x1280 (9:16 aspect ratio)")
    print("• Format: MP4 with H.264 video and AAC audio")
    print("• Optimized for: Instagram Reels, TikTok, YouTube Shorts")
    print("• Smart cropping: Preserves faces and important objects")
    
    print("\n🎯 FEATURES:")
    print("• 5 main categories with 20+ subcategories")
    print("• Mood-based search (Energetic, Calm, Professional, etc.)")
    print("• Duration filtering (Short, Medium, Long)")
    print("• Style options (Professional, Creative, Social, Cinematic)")
    print("• Preview before conversion")
    print("• Batch processing")
    
    print("\n🔧 SYSTEM REQUIREMENTS:")
    print("• FFmpeg (for video processing)")
    print("• Python 3.8+")
    print("• OpenAI API key (for AI agents)")
    print("• Internet connection (for Pexels API)")
    
    print("\n💡 TIPS:")
    print("• Use specific keywords for better results")
    print("• Preview videos before converting to save time")
    print("• Always credit photographers when posting content")
    print("• Check video duration - shorter content often performs better")
    
    print("\n🆘 TROUBLESHOOTING:")
    print("• 'FFmpeg not found': Install FFmpeg using package manager")
    print("• 'No videos found': Try different or broader search terms")
    print("• 'API error': Check internet connection and API keys")
    print("• 'Processing failed': Ensure sufficient disk space")

def display_results_summary(results: List[Dict]):
    """Display summary of conversion results"""
    print("\n" + "="*60)
    print("📊 CONVERSION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n📹 Video {i}:")
        print(f"  📄 File: {os.path.basename(result['output_file'])}")
        print(f"  📐 Resolution: {result['processing_result']['final_resolution']}")
        print(f"  🎭 By: {result['original_video']['photographer']}")
        print(f"  ⏱️  Duration: {result['original_video']['duration']}s")
        
        if os.path.exists(result['output_file']):
            file_size = os.path.getsize(result['output_file']) / (1024 * 1024)
            print(f"  💾 Size: {file_size:.1f} MB")
            print(f"  ✅ Ready for upload!")
    
    print(f"\n📁 All files saved in: output_reels/")
    print("🎉 Ready for Instagram Reels, TikTok, YouTube Shorts!")

def reinstall_dependencies():
    """Reinstall Python dependencies"""
    print("\n🔄 Reinstalling dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"], 
                      check=True)
        print("✅ Dependencies reinstalled successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to reinstall dependencies")

def clean_output_folder():
    """Clean the output folder"""
    output_dir = Path("output_reels")
    if not output_dir.exists():
        print("📁 No output folder to clean")
        return
    
    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        print("📹 No videos to clean")
        return
    
    confirm = input(f"🗑️  Delete {len(video_files)} video(s)? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        for video_file in video_files:
            video_file.unlink()
        print(f"✅ Cleaned {len(video_files)} video(s)")
    else:
        print("❌ Cleaning cancelled")

def export_configuration():
    """Export current configuration"""
    config = {
        "system": {
            "python_version": sys.version.split()[0],
            "working_directory": os.getcwd(),
            "ffmpeg_available": True,  # We checked this earlier
        },
        "api_keys": {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "pexels_configured": True,
        },
        "directories": {
            "output_dir": "output_reels",
            "logs_dir": "logs",
        }
    }
    
    config_file = Path("config_export.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration exported to: {config_file}")

def main():
    """Main function to run the application"""
    # Check if running from correct directory
    if not Path("video_reel_converter.py").exists():
        print("❌ Please run from the project directory containing video_reel_converter.py")
        sys.exit(1)
    
    # Initial setup
    print("🚀 AI Video Reel Generator - Starting up...")
    
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please fix the issues above.")
        sys.exit(1)
    
    setup_environment()
    
    # Main loop
    while True:
        try:
            display_main_menu()
            choice = input("\n🎯 Select an option (0-6): ").strip()
            
            if choice == "1":
                interactive_mode()
            elif choice == "2":
                quick_mode()
            elif choice == "3":
                demo_mode()
            elif choice == "4":
                settings_mode()
            elif choice == "5":
                view_results()
            elif choice == "6":
                help_mode()
            elif choice == "0":
                print("\n👋 Thanks for using AI Video Reel Generator!")
                print("🎬 Keep creating amazing content!")
                break
            else:
                print("❌ Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("🔄 Returning to main menu...")

if __name__ == "__main__":
    # Support command line arguments
    parser = argparse.ArgumentParser(description="AI Video Reel Generator")
    parser.add_argument("--mode", choices=["interactive", "quick", "demo"], 
                       help="Run in specific mode")
    parser.add_argument("--query", help="Search query for quick mode")
    parser.add_argument("--count", type=int, default=2, help="Number of videos for quick mode")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Direct quick mode
        if not check_dependencies():
            sys.exit(1)
        setup_environment()
        
        query = args.query or input("Enter search query: ")
        print(f"🚀 Quick mode: '{query}' ({args.count} videos)")
        
        try:
            from video_reel_converter import VideoReelConverter
            converter = VideoReelConverter("D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t")
            results = converter.convert_to_reel(query, per_page=args.count)
            if results:
                display_results_summary(results)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif args.mode == "interactive":
        # Direct interactive mode
        if not check_dependencies():
            sys.exit(1)
        setup_environment()
        interactive_mode()
    
    elif args.mode == "demo":
        # Direct demo mode
        demo_mode()
    
    else:
        # Regular main menu
        main()