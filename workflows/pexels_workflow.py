"""
Pexels Stock Workflow - AI-DRIVEN
One input from user → AI decides everything → Professional reel ready!
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.ai_agent_service import AIAgentService
from services.submagic_service import SubmagicService
from video_reel_converter import (
    PexelsVideoSearchTool,
    FalMusicGenerationTool,
    FalTTSGenerationTool,
    VideoInfo
)
from utils.video_mixer import VideoMixer
from utils.file_uploader import FileUploader
from config import Config
import logging

logger = logging.getLogger(__name__)


class PexelsWorkflow:
    """
    AI-Driven Pexels Stock Workflow

    Flow:
    1. User provides idea
    2. AI Agent (Claude + GPT-3.5) makes all decisions
    3. Search Pexels with AI keywords
    4. Download videos
    5. Generate music (AI-chosen style)
    6. Generate voice narration (AI-chosen voice/emotion/language)
    7. Mix videos + music + narration
    8. Upload to Supabase
    9. Submit to Submagic for subtitles + effects
    10. Download final reel
    """

    def __init__(self):
        """Initialize workflow with all required services"""
        # AI Agent
        self.ai_agent = AIAgentService()

        # Pexels
        self.pexels_tool = PexelsVideoSearchTool(Config.PEXELS_API_KEY)

        # Audio generation
        self.music_tool = FalMusicGenerationTool(Config.FAL_AI_API_KEY)
        self.tts_tool = FalTTSGenerationTool(Config.FAL_AI_API_KEY)

        # Video mixing
        self.video_mixer = VideoMixer(output_dir=Config.OUTPUT_DIR)

        # File upload
        self.file_uploader = FileUploader()

        # Submagic editing
        self.submagic = SubmagicService()

        # Output directory
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    def execute(self, user_idea: str, output_filename: str = None) -> dict:
        """
        Execute the complete AI-driven Pexels workflow

        Args:
            user_idea: User's video idea (e.g., "calming morning coffee video")
            output_filename: Optional custom output filename

        Returns:
            dict: {
                "success": bool,
                "output_file": str,
                "decisions": dict,
                "supabase_url": str,
                "error": str (if failed)
            }
        """

        try:
            print("\n" + "="*80)
            print("🎨 PEXELS STOCK WORKFLOW - AI-DRIVEN")
            print("="*80)
            print(f"💡 User Idea: '{user_idea}'")
            print()

            # ============================================================
            # STEP 1: AI AGENT MAKES ALL DECISIONS
            # ============================================================
            print("🤖 STEP 1: AI Agent analyzing your idea...")
            print("-" * 60)

            decisions = self.ai_agent.analyze_idea(user_idea)
            self.ai_agent.validate_decisions(decisions)

            print("\n✅ AI Decisions Complete!")
            print(f"   📂 Category: {decisions['category']}")
            print(f"   🔍 Keywords: {', '.join(decisions['keywords'])}")
            print(f"   🎵 Music: {decisions['music_style']}")
            print(f"   🎤 Voice: {decisions['voice_id']} ({decisions['emotion']})")
            print(f"   🌍 Language: {decisions['language']}")
            print()

            # ============================================================
            # STEP 2: GENERATE VOICE NARRATION FIRST (to get duration)
            # ============================================================
            print("🎤 STEP 2: Generating voice narration...")
            print("-" * 60)

            narration_text = decisions['narration']
            voice_id = decisions['voice_id']
            emotion = decisions['emotion']
            language = decisions['language']

            print(f"   Text: '{narration_text[:60]}...'")
            print(f"   Voice: {voice_id}")
            print(f"   Emotion: {emotion}")
            print(f"   Language: {language}")

            tts_result = self.tts_tool._run(
                text=narration_text,
                voice_id=voice_id,
                emotion=emotion,
                language=language
            )
            tts_data = json.loads(tts_result)

            if not tts_data.get("success"):
                print(f"   ⚠️  TTS generation failed: {tts_data.get('error')}")
                tts_url = None
                audio_duration = 20.0  # Fallback duration
            else:
                tts_url = tts_data.get("audio_url")
                print(f"   ✅ Voice narration generated: {tts_url[:60]}...")

                # Download audio and measure ACTUAL duration using ffprobe
                # (MiniMax API's duration_ms field is unreliable)
                print(f"   📏 Measuring actual audio duration...")
                try:
                    import tempfile
                    import subprocess
                    import requests

                    # Download audio file
                    temp_audio = Path(tempfile.mkdtemp()) / "temp_audio.mp3"
                    audio_response = requests.get(tts_url)
                    audio_response.raise_for_status()
                    with open(temp_audio, 'wb') as f:
                        f.write(audio_response.content)

                    # Get actual duration using ffprobe
                    cmd = [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(temp_audio)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0 and result.stdout.strip():
                        audio_duration = float(result.stdout.strip())
                        print(f"   ✅ Actual audio duration: {audio_duration:.1f}s (measured with ffprobe)")
                    else:
                        # Fallback to API duration
                        audio_duration = tts_data.get("duration", 20.0)
                        print(f"   ⚠️  Could not measure duration, using API value: {audio_duration:.1f}s")

                    # Cleanup temp file
                    temp_audio.unlink(missing_ok=True)

                except Exception as e:
                    # Fallback to API duration if measurement fails
                    audio_duration = tts_data.get("duration", 20.0)
                    print(f"   ⚠️  Duration measurement failed: {e}")
                    print(f"   Using API reported duration: {audio_duration:.1f}s")
            print()

            # ============================================================
            # STEP 3: CALCULATE VIDEO COUNT & DYNAMIC SEGMENT DURATION
            # ============================================================
            import math

            # Target 5 videos for variety, but adjust segment duration to match audio
            TARGET_NUM_VIDEOS = 5

            # Calculate segment duration dynamically: audio_duration / num_videos
            # This ensures video duration EXACTLY matches audio duration
            segment_duration = audio_duration / TARGET_NUM_VIDEOS
            num_videos_needed = TARGET_NUM_VIDEOS

            print(f"📊 STEP 3: Calculating video requirements...")
            print("-" * 60)
            print(f"   Audio duration: {audio_duration:.1f}s")
            print(f"   Target videos: {TARGET_NUM_VIDEOS} clips")
            print(f"   Segment duration: {segment_duration:.2f}s (dynamic)")
            print(f"   Total video length: {num_videos_needed * segment_duration:.1f}s")
            print(f"   ✅ Video matches audio duration perfectly!")
            print()

            # ============================================================
            # STEP 4: SEARCH PEXELS WITH AI KEYWORDS (WITH RETRY)
            # ============================================================
            print(f"🔍 STEP 4: Searching Pexels for {num_videos_needed} videos...")
            print("-" * 60)

            # Join keywords for search query
            search_query = " ".join(decisions['keywords'][:3])  # Use top 3 keywords
            print(f"   Search query: '{search_query}'")

            # Retry logic for Pexels API (handle 522 errors)
            import time
            max_retries = 3
            videos = None

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"   Attempt {attempt}/{max_retries}...")
                    # Search Pexels (fetch more than needed for variety)
                    pexels_result = self.pexels_tool._run(search_query, per_page=num_videos_needed + 2)

                    # PexelsVideoSearchTool returns a JSON-serialized list of videos
                    videos = json.loads(pexels_result)

                    # If it's a string error message, not a list
                    if isinstance(videos, str):
                        raise Exception(f"Pexels search failed: {videos}")

                    # If it's empty
                    if not videos or len(videos) == 0:
                        raise Exception("No videos found for search query")

                    print(f"   ✅ Found {len(videos)} videos")
                    break  # Success, exit retry loop

                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        print(f"   ⚠️  Pexels API error (attempt {attempt}/{max_retries}), retrying in 3s...")
                        time.sleep(3)
                    else:
                        raise Exception(f"Pexels API failed after {max_retries} attempts: {e}")
                except Exception as e:
                    if "522" in str(e) or "Error searching" in str(e):
                        if attempt < max_retries:
                            print(f"   ⚠️  Pexels server error (attempt {attempt}/{max_retries}), retrying in 5s...")
                            time.sleep(5)
                        else:
                            raise Exception(f"Pexels API unavailable after {max_retries} attempts")
                    else:
                        raise  # Re-raise other exceptions immediately

            if not videos:
                raise Exception("Failed to get videos from Pexels after retries")

            print()

            # ============================================================
            # STEP 5: GENERATE MUSIC (AI-CHOSEN STYLE)
            # ============================================================
            print("🎵 STEP 5: Generating background music...")
            print("-" * 60)

            # Create music prompt from AI decision
            music_prompt = f"{decisions['music_style']} music for {decisions['category']}"
            print(f"   Music prompt: '{music_prompt}'")
            print(f"   Duration needed: {audio_duration:.1f}s")

            # FalMusicGenerationTool only takes prompt_text parameter
            music_result = self.music_tool._run(prompt_text=music_prompt)
            music_data = json.loads(music_result)

            if not music_data.get("success"):
                print(f"   ⚠️  Music generation failed: {music_data.get('error')}")
                music_url = None
            else:
                music_url = music_data.get("audio_url")
                print(f"   ✅ Music generated: {music_url[:60]}...")
            print()

            # ============================================================
            # STEP 6: MIX VIDEOS + MUSIC + NARRATION
            # ============================================================
            print("🎬 STEP 6: Mixing videos with audio...")
            print("-" * 60)

            # Extract video download URLs from Pexels results
            video_download_urls = []
            for video in videos[:num_videos_needed]:  # Use exact count needed
                download_url = video.get('download_url')
                if download_url:
                    video_download_urls.append(download_url)

            if not video_download_urls:
                raise Exception("No video download URLs found")

            print(f"   Processing {len(video_download_urls)} videos...")
            print(f"   Each video: {segment_duration:.2f}s (dynamic)")
            print(f"   Total video: {len(video_download_urls) * segment_duration:.1f}s")
            print(f"   Audio length: {audio_duration:.1f}s")

            # Create reel with VideoMixer
            from datetime import datetime
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"ai_reel_{timestamp}.mp4"

            mixed_video_path = self.video_mixer.create_reel(
                video_urls=video_download_urls,
                music_url=music_url,
                voice_url=tts_url,
                segment_duration=segment_duration,  # Dynamic duration matching audio
                output_filename=output_filename
            )

            print(f"   ✅ Reel created: {Path(mixed_video_path).name}")
            file_size_mb = Path(mixed_video_path).stat().st_size / (1024 * 1024)
            print(f"   📦 Size: {file_size_mb:.2f} MB")
            print()

            # ============================================================
            # STEP 7: UPLOAD TO SUPABASE
            # ============================================================
            print("☁️  STEP 7: Uploading to Supabase...")
            print("-" * 60)

            try:
                supabase_url = self.file_uploader.upload(mixed_video_path)
                print(f"   ✅ Upload successful!")
                print(f"   🔗 URL: {supabase_url[:80]}...")
            except Exception as e:
                print(f"   ⚠️  Upload failed: {e}")
                supabase_url = None

            print()

            # ============================================================
            # STEP 8: SUBMIT TO SUBMAGIC (SUBTITLES + EFFECTS)
            # ============================================================
            print("✨ STEP 8: Submitting to Submagic...")
            print("-" * 60)

            final_video_url = None
            final_output_path = None

            if not supabase_url:
                print("   ⚠️  Skipping Submagic: No Supabase URL available")
            else:
                try:
                    # Use complete workflow method for Submagic processing
                    from datetime import datetime
                    final_filename = f"final_with_subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    final_output_path = self.output_dir / final_filename

                    print(f"   Submitting video URL: {supabase_url[:60]}...")
                    print("   Settings: Iman template, Magic Zooms, Magic B-rolls (60%)")

                    # Complete end-to-end Submagic workflow
                    final_path = self.submagic.process_video(
                        video_url=supabase_url,
                        output_path=str(final_output_path),
                        title="AI Generated Reel",
                        language=decisions['language'][:2].lower(),  # "en", "hi", "ar"
                        template_name="Iman",
                        magic_zooms=True,
                        magic_brolls=True,
                        magic_brolls_percentage=60
                    )

                    print(f"   ✅ Submagic processing complete!")
                    print(f"   📥 Downloaded: {final_filename}")

                    # Update path reference
                    final_output_path = Path(final_path)
                    final_video_url = "processed"  # Mark as successful

                except Exception as e:
                    print(f"   ⚠️  Submagic processing failed: {e}")
                    final_video_url = None
                    final_output_path = None
                    logger.warning(f"Submagic failed, using video without subtitles: {e}")

            print()

            # ============================================================
            # RETURN RESULTS
            # ============================================================
            print("="*80)
            print("✅ WORKFLOW EXECUTION SUMMARY")
            print("="*80)
            print(f"✅ AI Decisions: Complete")
            print(f"✅ Pexels Search: {len(videos)} videos found")
            print(f"✅ Music Generation: {'Complete' if music_url else 'Failed'}")
            print(f"✅ Voice Narration: {'Complete' if tts_url else 'Failed'}")
            print(f"✅ Video Mixing: {'Complete' if mixed_video_path else 'Failed'}")
            print(f"✅ Supabase Upload: {'Complete' if supabase_url else 'Failed'}")
            print(f"✅ Submagic Processing: {'Complete' if final_video_url else 'Failed'}")
            print()
            if mixed_video_path:
                print(f"📁 Mixed video (no subtitles): {mixed_video_path}")
            if final_output_path:
                print(f"📁 Final video (with subtitles): {final_output_path}")
            if supabase_url:
                print(f"🔗 Public URL: {supabase_url}")
            if final_video_url:
                print(f"🔗 Final video URL: {final_video_url}")
            print("="*80)

            # Determine overall status
            if final_video_url and final_output_path:
                status = "success"
                message = "Complete! Video with AI-generated subtitles ready."
            elif mixed_video_path and supabase_url:
                status = "partial_success"
                message = "Reel created and uploaded. Submagic processing failed."
            elif mixed_video_path:
                status = "partial_success"
                message = "Reel created locally. Upload failed - use local file."
            else:
                status = "partial_success"
                message = "Audio generated. Video mixing incomplete."

            return {
                "success": True,
                "status": status,
                "message": message,
                "decisions": decisions,
                "pexels_videos": videos,
                "music_url": music_url,
                "tts_url": tts_url,
                "mixed_video_path": mixed_video_path,
                "supabase_url": supabase_url,
                "final_video_url": final_video_url,
                "final_output_path": str(final_output_path) if final_output_path else None
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# ============================================================
# TEST/DEMO
# ============================================================
if __name__ == "__main__":
    print("🧪 Testing Pexels AI-Driven Workflow\n")

    # Test ideas
    test_ideas = [
        "I want a calming video about morning coffee routines",
        "Create an energetic workout motivation reel",
        "मुझे योग के बारे में एक peaceful वीडियो चाहिए"
    ]

    workflow = PexelsWorkflow()

    for i, idea in enumerate(test_ideas, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_ideas)}")
        print(f"{'='*80}\n")

        result = workflow.execute(idea)

        if result["success"]:
            print(f"\n✅ Test {i} PASSED")
        else:
            print(f"\n❌ Test {i} FAILED: {result.get('error')}")

        input("\nPress Enter to continue to next test...")
