#!/usr/bin/env python3
"""
Test script for the new logo overlay feature
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

def create_test_logo(logo_path: str = "test_logo.png"):
    """Create a simple test logo for testing"""
    try:
        # Create a simple 200x200 transparent logo with text
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a circle background
        draw.ellipse([10, 10, 190, 190], fill=(255, 100, 100, 180))
        
        # Add text
        try:
            # Try to use a system font
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Center the text
        text = "TEST"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (200 - text_width) // 2
        y = (200 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        
        # Save the logo
        img.save(logo_path, "PNG")
        print(f"✅ Test logo created: {logo_path}")
        return logo_path
        
    except Exception as e:
        print(f"❌ Failed to create test logo: {e}")
        return None

def test_logo_feature():
    """Test the logo overlay feature"""
    print("🧪 Testing Logo Overlay Feature")
    print("=" * 50)
    
    # Create a test logo if it doesn't exist
    logo_path = "test_logo.png"
    current_dir = os.getcwd()
    full_logo_path = os.path.join(current_dir, logo_path)
    
    if not os.path.exists(full_logo_path):
        created_logo = create_test_logo(full_logo_path)
        if not created_logo:
            print("❌ Cannot create test logo. Please provide your own logo file.")
            return False
    
    print(f"📁 Current directory: {current_dir}")
    print(f"🖼️  Test logo location: {full_logo_path}")
    print(f"✅ Logo exists: {os.path.exists(full_logo_path)}")
    
    # Test the interactive generator with logo
    print("\n🚀 You can now run the interactive generator and test with this logo path:")
    print(f"   {full_logo_path}")
    print("\n💡 To test:")
    print("1. Run: python interactive_reel_generator.py")
    print("2. When prompted for logo path, enter:")
    print(f"   {full_logo_path}")
    print("3. Complete the other preferences and generate a reel")
    print("4. Check if the logo appears in the top-right corner")
    
    return True

if __name__ == "__main__":
    test_logo_feature()