#!/usr/bin/env python3
"""
Test script for Local Media Workflow with Wedding Planner theme
Tests logo overlay functionality with logo.jpg
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from workflows.local_media_workflow import LocalMediaWorkflow

def main():
    print("=" * 80)
    print("🎬 LOCAL MEDIA WORKFLOW TEST - WEDDING PLANNER")
    print("=" * 80)

    # Configuration
    video_idea = "wedding planner ki professional services showcase karte hue, shaadi ki planning aur decoration"
    video_directory = input("\n📂 Enter path to video directory: ").strip()

    if not video_directory:
        print("❌ No directory path provided.")
        return

    if not os.path.isdir(video_directory):
        print(f"❌ Invalid directory path: {video_directory}")
        return

    # Logo path (main directory)
    logo_path = str(Path(__file__).parent / "logo.jpg")

    if not os.path.exists(logo_path):
        print(f"⚠️  Logo not found at: {logo_path}")
        logo_path = None
    else:
        print(f"✅ Logo found: {logo_path}")
        print(f"   Logo will be placed at bottom-right corner")

    print(f"\n💡 Video Idea: '{video_idea}'")
    print(f"📁 Video Directory: {video_directory}")
    print(f"🖼️  Logo: {'Yes (logo.jpg)' if logo_path else 'No'}")

    confirm = input("\n🚀 Start workflow? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Cancelled.")
        return

    # Execute workflow
    try:
        workflow = LocalMediaWorkflow()
        result = workflow.execute(
            user_idea=video_idea,
            video_directory=video_directory,
            logo_path=logo_path
        )

        if result["success"]:
            print("\n" + "=" * 80)
            print("✅ WORKFLOW COMPLETED!")
            print("=" * 80)
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            print(f"Videos processed: {result.get('num_videos')}")
            print(f"Clip duration: {result.get('clip_duration', 0):.2f}s each")

            if result.get('mixed_video_path'):
                print(f"\n📁 Mixed video (with logo, no subtitles): {result.get('mixed_video_path')}")

            if result.get('final_output_path'):
                print(f"📁 Final video (with logo + subtitles): {result.get('final_output_path')}")

            if result.get('music_url'):
                print(f"🎵 Music: Generated")

            if result.get('tts_url'):
                print(f"🎤 Voice: Generated")

            print("\n💡 TIP: Check if logo appears in bottom-right corner!")

        else:
            print(f"\n❌ Workflow failed: {result.get('error')}")
            if result.get('traceback'):
                print("\nTraceback:")
                print(result.get('traceback'))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
