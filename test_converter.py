#!/usr/bin/env python3
"""
Simple test script for the Video Reel Converter
"""

import os
from dotenv import load_dotenv
from video_reel_converter import VideoReelConverter

# Load environment variables
load_dotenv()

def test_converter():
    """Test the video reel converter with a simple query"""
    
    # Pexels API key
    PEXELS_API_KEY = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
    
    # Check if OpenAI API key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("💡 Make sure your .env file contains: OPENAI_API_KEY='your-key-here'")
        return
    
    print("🚀 Starting Video Reel Converter Test")
    print("="*50)
    
    # Initialize converter
    try:
        converter = VideoReelConverter(PEXELS_API_KEY)
        print("✅ Converter initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing converter: {e}")
        return
    
    try:
        # Test with a simple query for 1 video
        query = "sunset ocean"
        print(f"🔍 Searching for videos with query: '{query}'")
        
        results = converter.convert_to_reel(query, per_page=1)
        
        if results:
            print(f"\n🎉 Success! Converted {len(results)} video(s)")
            
            for i, result in enumerate(results, 1):
                print(f"\nVideo {i}:")
                print(f"  📁 Output file: {result['output_file']}")
                print(f"  📐 Original size: {result['original_video']['width']}x{result['original_video']['height']}")
                print(f"  ⏱️  Duration: {result['original_video']['duration']} seconds")
                print(f"  🎭 Photographer: {result['original_video']['photographer']}")
                print(f"  🎯 Objects detected: {result['detection_result']['detections_count']}")
                print(f"  🔄 Smart cropping: {'Yes' if result['detection_result']['roi_detected'] else 'Center crop'}")
                print(f"  📱 Final resolution: {result['processing_result']['final_resolution']}")
                
                # Check if file exists
                if os.path.exists(result['output_file']):
                    file_size = os.path.getsize(result['output_file']) / (1024 * 1024)  # MB
                    print(f"  💾 File size: {file_size:.1f} MB")
                    print(f"  ✅ File created successfully")
                else:
                    print(f"  ❌ Output file not found")
        else:
            print("❌ No videos were processed")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
    
    finally:
        # Cleanup
        try:
            converter.cleanup()
            print("\n🧹 Temporary files cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Cleanup error: {e}")
    
    print("\n" + "="*50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_converter()