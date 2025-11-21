"""
Test script for Submagic API integration
Processes the video that's already uploaded to Supabase
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.submagic_service import SubmagicService
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test Submagic subtitle generation on the uploaded video"""

    # Your video URL from the previous workflow
    video_url = "https://pfixecrevydvermbgwmo.supabase.co/storage/v1/object/sign/reel-videos/20251121_171403_ai_reel_20251121_171343.mp4?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV9lYzgzNjg4MS01Nzc4LTQwMWQtYjRhMi1iOWJjOGYxYzgyOTgiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJyZWVsLXZpZGVvcy8yMDI1MTEyMV8xNzE0MDNfYWlfcmVlbF8yMDI1MTEyMV8xNzEzNDMubXA0IiwiaWF0IjoxNzYzNzI1NDQ1LCJleHAiOjE3NjM3MjkwNDV9.zVo-YVpPzC7pyAhBBCpXlLuGdRBD4VECmekLbgKlCyo"

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/ai_reel_with_subtitles_{timestamp}.mp4"

    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("🎬 SUBMAGIC SUBTITLE GENERATION TEST")
    print("=" * 80)
    print(f"\n📹 Video URL: {video_url[:70]}...")
    print(f"💾 Output: {output_path}")
    print(f"🌐 Language: English")
    print(f"🎨 Template: Iman")
    print(f"✨ Magic Zooms: Enabled")
    print(f"🎬 Magic B-rolls: Enabled (60%)")
    print(f"⏰ Timeout: {Config.SUBMAGIC_TIMEOUT} seconds")
    print("\n" + "=" * 80 + "\n")

    try:
        # Initialize service
        logger.info("Initializing Submagic service...")
        service = SubmagicService()
        logger.info("✅ Service initialized successfully\n")

        # Process video (complete workflow)
        logger.info("Starting Submagic processing...")
        final_video = service.process_video(
            video_url=video_url,
            output_path=output_path,
            title="AI Generated Reel with Subtitles",
            language="en",
            template_name="Iman",
            magic_zooms=True,
            magic_brolls=True,
            magic_brolls_percentage=60
        )

        print("\n" + "=" * 80)
        print("✅ SUBMAGIC PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\n📁 Final video saved: {final_video}")
        print(f"✨ Video includes:")
        print(f"   • Auto-generated subtitles (98%+ accuracy)")
        print(f"   • Professional caption styling (Iman template)")
        print(f"   • AI Magic Zooms (auto-zoom effects)")
        print(f"   • AI Magic B-rolls (60% density)")
        print(f"   • Social media optimized (9:16)")
        print("\n" + "=" * 80 + "\n")

        return final_video

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ SUBMAGIC PROCESSING FAILED")
        print("=" * 80)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()

        print("\n" + "=" * 80 + "\n")
        return None


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
