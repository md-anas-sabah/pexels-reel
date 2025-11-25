"""
Local Media Workflow - AI-DRIVEN
User's local videos + AI-generated audio → Professional reel with subtitles

Flow:
1. User provides video idea
2. AI Agent (Claude + GPT-3.5) makes decisions (narration, music, voice, emotion)
3. Scan local directory for video files
4. Trim videos with DYNAMIC duration (20s / number of clips)
5. Stitch/concatenate videos
6. Generate audio (music + voice narration via Fal AI)
7. Mix audio into video
8. Upload to Supabase → get public URL
9. Send to Submagic for subtitles + effects
10. Download final output
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.ai_agent_service import AIAgentService
from services.submagic_service import SubmagicService
from utils.file_uploader import FileUploader
from config import Config

logger = logging.getLogger(__name__)


class LocalMediaWorkflow:
    """
    AI-Driven Local Media Workflow

    Takes user's local video files and creates a professional reel with:
    - AI-generated narration and music
    - Dynamic clip trimming (20s total / number of clips)
    - Professional subtitles via Submagic
    """

    # Target total video duration (seconds)
    TARGET_TOTAL_DURATION = 20.0

    def __init__(self):
        """Initialize workflow with all required services"""
        # AI Agent for decisions
        self.ai_agent = AIAgentService()

        # File uploader (Supabase)
        self.file_uploader = FileUploader()

        # Submagic for subtitles
        self.submagic = SubmagicService()

        # Output directory
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

        # Supported video formats
        self.supported_formats = Config.SUPPORTED_VIDEO_FORMATS

    def scan_directory(self, directory_path: str) -> List[str]:
        """
        Scan directory for video files

        Args:
            directory_path: Path to directory containing videos

        Returns:
            List of video file paths (sorted alphabetically)
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        video_files = []

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                video_files.append(str(file_path))

        # Sort alphabetically for consistent ordering
        video_files.sort()

        if not video_files:
            raise ValueError(
                f"No video files found in {directory_path}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        return video_files

    def calculate_clip_duration(self, num_clips: int, target_total: float = None) -> float:
        """
        Calculate duration per clip based on number of clips

        Formula: duration_per_clip = target_total / num_clips

        Examples:
            - 4 videos = 20s / 4 = 5s each
            - 5 videos = 20s / 5 = 4s each
            - 3 videos = 20s / 3 = 6.67s each

        Args:
            num_clips: Number of video clips
            target_total: Target total duration (default: 20s)

        Returns:
            Duration per clip in seconds
        """
        if target_total is None:
            target_total = self.TARGET_TOTAL_DURATION

        if num_clips <= 0:
            raise ValueError("Number of clips must be positive")

        duration = target_total / num_clips

        logger.info(f"📊 Dynamic clip duration: {target_total}s / {num_clips} clips = {duration:.2f}s each")

        return duration

    def execute(self, user_idea: str, video_directory: str, output_filename: str = None, logo_path: str = None) -> dict:
        """
        Execute the complete Local Media workflow

        Args:
            user_idea: User's video idea for AI to generate audio
            video_directory: Path to directory containing local videos
            output_filename: Optional custom output filename
            logo_path: Optional path to logo image for overlay (bottom-right corner)

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
            print("\n" + "=" * 80)
            print("🎬 LOCAL MEDIA WORKFLOW - AI-DRIVEN")
            print("=" * 80)
            print(f"💡 User Idea: '{user_idea}'")
            print(f"📁 Video Directory: {video_directory}")
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
            print(f"   🎵 Music: {decisions['music_style']}")
            print(f"   🎤 Voice: {decisions['voice_id']} ({decisions['emotion']})")
            print(f"   🌍 Language: {decisions['language']}")
            print()

            # ============================================================
            # STEP 2: SCAN LOCAL DIRECTORY FOR VIDEOS
            # ============================================================
            print("📁 STEP 2: Scanning local directory for videos...")
            print("-" * 60)

            video_files = self.scan_directory(video_directory)
            num_videos = len(video_files)

            print(f"   ✅ Found {num_videos} video(s):")
            for i, vf in enumerate(video_files, 1):
                print(f"      {i}. {Path(vf).name}")
            print()

            # ============================================================
            # STEP 3: GENERATE VOICE NARRATION (to get duration)
            # ============================================================
            print("🎤 STEP 3: Generating voice narration...")
            print("-" * 60)

            # Import audio tools from video_reel_converter
            from video_reel_converter import FalMusicGenerationTool, FalTTSGenerationTool

            tts_tool = FalTTSGenerationTool(Config.FAL_AI_API_KEY)

            narration_text = decisions['narration']
            voice_id = decisions['voice_id']
            emotion = decisions['emotion']
            language = decisions['language']

            print(f"   Text: '{narration_text[:60]}...'")
            print(f"   Voice: {voice_id}")
            print(f"   Emotion: {emotion}")
            print(f"   Language: {language}")

            tts_result = tts_tool._run(
                text=narration_text,
                voice_id=voice_id,
                emotion=emotion,
                language=language
            )
            tts_data = json.loads(tts_result)

            if not tts_data.get("success"):
                print(f"   ⚠️  TTS generation failed: {tts_data.get('error')}")
                tts_url = None
                audio_duration = self.TARGET_TOTAL_DURATION  # Fallback
            else:
                tts_url = tts_data.get("audio_url")
                print(f"   ✅ Voice narration generated!")

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
                        audio_duration = tts_data.get("duration", self.TARGET_TOTAL_DURATION)
                        print(f"   ⚠️  Could not measure duration, using API value: {audio_duration:.1f}s")

                    # Cleanup temp file
                    temp_audio.unlink(missing_ok=True)

                except Exception as e:
                    # Fallback to API duration if measurement fails
                    audio_duration = tts_data.get("duration", self.TARGET_TOTAL_DURATION)
                    print(f"   ⚠️  Duration measurement failed: {e}")
                    print(f"   Using API reported duration: {audio_duration:.1f}s")
            print()

            # ============================================================
            # STEP 4: CALCULATE DYNAMIC CLIP DURATION
            # ============================================================
            print("📊 STEP 4: Calculating dynamic clip duration...")
            print("-" * 60)

            # Use audio duration as target total (so video matches audio)
            target_duration = audio_duration if audio_duration else self.TARGET_TOTAL_DURATION
            clip_duration = self.calculate_clip_duration(num_videos, target_duration)

            print(f"   Target total: {target_duration:.1f}s")
            print(f"   Number of clips: {num_videos}")
            print(f"   Duration per clip: {clip_duration:.2f}s")
            print(f"   Final video length: {num_videos * clip_duration:.1f}s")
            print()

            # ============================================================
            # STEP 5: GENERATE BACKGROUND MUSIC
            # ============================================================
            print("🎵 STEP 5: Generating background music...")
            print("-" * 60)

            music_tool = FalMusicGenerationTool(Config.FAL_AI_API_KEY)

            music_prompt = f"{decisions['music_style']} music for {decisions['category']}"
            print(f"   Music prompt: '{music_prompt}'")

            music_result = music_tool._run(prompt_text=music_prompt)
            music_data = json.loads(music_result)

            if not music_data.get("success"):
                print(f"   ⚠️  Music generation failed: {music_data.get('error')}")
                music_url = None
            else:
                music_url = music_data.get("audio_url")
                print(f"   ✅ Music generated!")
            print()

            # ============================================================
            # STEP 6: TRIM, CONCATENATE AND MIX AUDIO
            # ============================================================
            print("🎬 STEP 6: Processing videos with audio...")
            print("-" * 60)

            from utils.video_mixer import VideoMixer

            # Create video mixer with temp directory
            video_mixer = VideoMixer(output_dir=str(self.output_dir))

            # For local files, we don't need to download - just use paths directly
            # But we need to modify VideoMixer to accept local paths

            # Trim and concatenate local videos
            print(f"   Trimming {num_videos} videos to {clip_duration:.2f}s each...")
            concat_video = video_mixer.trim_and_concat_videos(
                video_paths=video_files,
                segment_duration=clip_duration
            )
            print(f"   ✅ Videos concatenated: {Path(concat_video).name}")

            # Mix audio with video
            from datetime import datetime
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"local_reel_{timestamp}.mp4"

            mixed_video_path = video_mixer.mix_audio_with_video(
                video_path=concat_video,
                music_url=music_url,
                voice_url=tts_url,
                output_filename=output_filename
            )

            print(f"   ✅ Audio mixed: {Path(mixed_video_path).name}")
            file_size_mb = Path(mixed_video_path).stat().st_size / (1024 * 1024)
            print(f"   📦 Size: {file_size_mb:.2f} MB")
            print()

            # ============================================================
            # STEP 6.5: ADD LOGO OVERLAY (if provided)
            # ============================================================
            if logo_path:
                print("🖼️  STEP 6.5: Adding logo overlay...")
                print("-" * 60)

                if not Path(logo_path).exists():
                    print(f"   ⚠️  Logo file not found: {logo_path}")
                    print("   Skipping logo overlay...")
                else:
                    try:
                        # Add logo overlay to mixed video
                        video_with_logo = video_mixer.add_logo_overlay(
                            video_path=mixed_video_path,
                            logo_path=logo_path
                        )

                        # Update mixed_video_path to use video with logo
                        mixed_video_path = video_with_logo

                        print(f"   ✅ Logo overlay added successfully!")
                        logo_file_size_mb = Path(mixed_video_path).stat().st_size / (1024 * 1024)
                        print(f"   📦 Size: {logo_file_size_mb:.2f} MB")
                    except Exception as e:
                        print(f"   ⚠️  Logo overlay failed: {e}")
                        print("   Continuing with video without logo...")
                print()
            else:
                print("🖼️  STEP 6.5: Logo overlay skipped (no logo provided)")
                print("-" * 60)
                print("   To add a logo, provide logo_path parameter")
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
                    final_filename = f"final_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    final_output_path = self.output_dir / final_filename

                    print(f"   Submitting video URL...")
                    print("   Settings: Iman template, Magic Zooms, NO B-rolls (using original videos)")

                    final_path = self.submagic.process_video(
                        video_url=supabase_url,
                        output_path=str(final_output_path),
                        title="AI Generated Local Reel",
                        language=decisions['language'][:2].lower(),
                        template_name="Iman",
                        magic_zooms=True,
                        magic_brolls=False,  # ❌ DISABLED - Keep original videos
                        magic_brolls_percentage=0  # No B-roll insertion
                    )

                    print(f"   ✅ Submagic processing complete!")
                    print(f"   📥 Downloaded: {final_filename}")

                    final_output_path = Path(final_path)
                    final_video_url = "processed"

                except Exception as e:
                    print(f"   ⚠️  Submagic processing failed: {e}")
                    final_video_url = None
                    final_output_path = None
                    logger.warning(f"Submagic failed: {e}")
            print()

            # ============================================================
            # STEP 9: CLEANUP
            # ============================================================
            video_mixer.cleanup()

            # ============================================================
            # RETURN RESULTS
            # ============================================================
            print("=" * 80)
            print("✅ WORKFLOW EXECUTION SUMMARY")
            print("=" * 80)
            print(f"✅ AI Decisions: Complete")
            print(f"✅ Local Videos: {num_videos} found")
            print(f"✅ Clip Duration: {clip_duration:.2f}s per video (dynamic)")
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
            print("=" * 80)

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
                "local_videos": video_files,
                "num_videos": num_videos,
                "clip_duration": clip_duration,
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
    import argparse

    parser = argparse.ArgumentParser(description="Local Media Workflow")
    parser.add_argument("--idea", type=str, required=True, help="Video idea for AI")
    parser.add_argument("--directory", type=str, required=True, help="Path to video directory")
    parser.add_argument("--output", type=str, default=None, help="Output filename")

    args = parser.parse_args()

    print("🧪 Testing Local Media Workflow\n")

    workflow = LocalMediaWorkflow()
    result = workflow.execute(
        user_idea=args.idea,
        video_directory=args.directory,
        output_filename=args.output
    )

    if result["success"]:
        print(f"\n✅ Workflow completed: {result['status']}")
        print(f"   Message: {result['message']}")
    else:
        print(f"\n❌ Workflow failed: {result.get('error')}")
