#!/usr/bin/env python3
"""
Demo script for Interactive Reel Generator
Simple demo to show the new interface
"""

import os
from interactive_reel_generator import ReelGeneratorUI

def show_features():
    """Show all the new features"""
    print("\n" + "="*80)
    print("🚀 NEW INTERACTIVE REEL GENERATOR FEATURES")
    print("="*80)
    
    print("\n📂 VIDEO CATEGORIES:")
    print("   1. Nature & Lifestyle (Ocean, Mountains, Forest, Sunset)")
    print("   2. Urban & City Life (Skyline, Streets, Architecture)")
    print("   3. People & Activities (Fitness, Food, Work, Lifestyle)")
    print("   4. Abstract & Creative (Colors, Lights, Textures)")
    print("   5. Seasonal & Weather (Spring, Summer, Autumn, Winter)")
    
    print("\n🎭 MOOD OPTIONS:")
    print("   • Energetic - Dynamic, fast-paced content")
    print("   • Calm & Peaceful - Serene, relaxing videos")
    print("   • Inspiring - Motivational, uplifting content")
    print("   • Professional - Corporate, business-ready")
    print("   • Creative - Artistic, unique visuals")
    
    print("\n⏱️  DURATION PREFERENCES:")
    print("   • Short (5-15 seconds) - Quick, punchy content")
    print("   • Medium (15-30 seconds) - Standard reel length")
    print("   • Long (30+ seconds) - Detailed, story-telling")
    print("   • Any Duration - No preference")
    
    print("\n🎨 STYLE OPTIONS:")
    print("   • Professional/Corporate - Clean, business-ready")
    print("   • Creative/Artistic - Unique, artistic feel")
    print("   • Social Media Ready - Optimized for engagement")
    print("   • Cinematic - Movie-like quality")
    
    print("\n✨ KEY FEATURES:")
    print("   📱 Perfect 9:16 aspect ratio for all platforms")
    print("   🎯 Smart object detection (faces, people, objects)")
    print("   🔍 Custom search options")
    print("   📋 Video preview and selection")
    print("   ⚙️  Duration filtering")
    print("   🎬 Multiple videos at once")
    print("   📁 Organized output folder")
    
    print("\n🎪 EXAMPLE USE CASES:")
    print("   • Instagram Reels for business")
    print("   • TikTok content creation")
    print("   • YouTube Shorts")
    print("   • Social media marketing")
    print("   • Personal content creation")
    
    print("\n" + "="*80)

def run_demo():
    """Run a simple demo"""
    show_features()
    
    choice = input("\n🚀 Want to try the interactive generator? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes', 'हां']:
        print("\n🎬 Starting Interactive Reel Generator...")
        generator = ReelGeneratorUI()
        generator.run()
    else:
        print("\n📝 Quick Start Commands:")
        print("   python interactive_reel_generator.py  # Full interactive mode")
        print("   python video_reel_converter.py        # Basic mode")
        print("   python test_converter.py              # Simple test")
        
        print("\n💡 Example Workflow:")
        print("   1. Select category (e.g., Nature & Lifestyle)")
        print("   2. Choose subcategory (e.g., Ocean & Beach)")
        print("   3. Pick mood (e.g., Calm & Peaceful)")
        print("   4. Set duration (e.g., Medium 15-30 seconds)")
        print("   5. Choose style (e.g., Social Media Ready)")
        print("   6. Preview and select videos")
        print("   7. Get perfectly cropped 720x1280 reels!")

if __name__ == "__main__":
    run_demo()