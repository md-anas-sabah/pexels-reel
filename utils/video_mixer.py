"""
Video Mixer Utility
Uses existing FFmpeg functions to mix videos with audio
"""

import os
import sys
import subprocess
import requests
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

logger = logging.getLogger(__name__)


class VideoMixer:
    """
    Video mixing utility that combines:
    - Multiple Pexels video clips
    - Background music
    - Voice narration
    Into a single professional reel
    """

    def __init__(self, output_dir: str = None):
        """Initialize video mixer"""
        self.output_dir = Path(output_dir or Config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_mixer_"))
        logger.info(f"Temp directory: {self.temp_dir}")

    def download_videos(self, video_urls: List[str]) -> List[str]:
        """
        Download videos from URLs

        Args:
            video_urls: List of video download URLs

        Returns:
            List of local file paths
        """
        downloaded_paths = []

        for i, url in enumerate(video_urls):
            try:
                logger.info(f"📥 Downloading video {i+1}/{len(video_urls)}...")

                response = requests.get(url, stream=True)
                response.raise_for_status()

                video_path = self.temp_dir / f"pexels_video_{i}.mp4"

                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded_paths.append(str(video_path))
                logger.info(f"✅ Downloaded: {video_path.name}")

            except Exception as e:
                logger.error(f"❌ Failed to download video {i+1}: {e}")

        return downloaded_paths

    def trim_and_concat_videos(
        self,
        video_paths: List[str],
        segment_duration: float = 5.0
    ) -> str:
        """
        Trim videos to equal length and concatenate them

        Args:
            video_paths: List of video file paths
            segment_duration: Duration of each segment in seconds

        Returns:
            Path to concatenated video (NO AUDIO)
        """
        if not video_paths:
            raise ValueError("No video paths provided")

        logger.info(f"✂️  Trimming {len(video_paths)} videos to {segment_duration}s each...")

        # Trim all videos
        trimmed_paths = []
        for i, video_path in enumerate(video_paths):
            trimmed_path = self.temp_dir / f"trimmed_{i}.mp4"

            if self._trim_segment(
                video_path,
                start_time=0,
                duration=segment_duration,
                output_path=str(trimmed_path)
            ):
                trimmed_paths.append(str(trimmed_path))

        if not trimmed_paths:
            raise Exception("Failed to trim any videos")

        logger.info(f"✅ Trimmed {len(trimmed_paths)} videos")

        # Concatenate trimmed videos
        logger.info("🔗 Concatenating videos...")
        concat_path = self.temp_dir / "concatenated_video.mp4"

        if self._concat_clips(trimmed_paths, str(concat_path)):
            logger.info(f"✅ Videos concatenated: {concat_path}")
            return str(concat_path)
        else:
            raise Exception("Failed to concatenate videos")

    def mix_audio_with_video(
        self,
        video_path: str,
        music_url: Optional[str] = None,
        voice_url: Optional[str] = None,
        output_filename: str = "final_reel.mp4"
    ) -> str:
        """
        Mix audio (music + voice) with video

        Args:
            video_path: Path to video file (no audio)
            music_url: URL of background music (optional)
            voice_url: URL of voice narration (optional)
            output_filename: Output filename

        Returns:
            Path to final mixed video
        """
        if not music_url and not voice_url:
            logger.warning("⚠️  No audio provided, returning original video")
            return video_path

        logger.info("🎵 Mixing audio with video...")

        output_path = self.output_dir / output_filename

        # Download audio files
        temp_audio_paths = []

        if voice_url:
            voice_path = self.temp_dir / "voice.mp3"
            self._download_audio(voice_url, str(voice_path))
            temp_audio_paths.append(("voice", str(voice_path)))

        if music_url:
            music_path = self.temp_dir / "music.mp3"
            self._download_audio(music_url, str(music_path))
            temp_audio_paths.append(("music", str(music_path)))

        # Build FFmpeg command
        cmd = ["ffmpeg", "-i", video_path]

        # Add audio inputs
        for _, audio_path in temp_audio_paths:
            cmd.extend(["-i", audio_path])

        # Audio mixing filter
        if len(temp_audio_paths) == 2:
            # Both voice and music
            logger.info("🎵 Mixing voice + music (music at 15% volume)")
            cmd.extend([
                "-filter_complex",
                "[2:a]volume=0.15[music_low];[1:a][music_low]amix=inputs=2:duration=first:dropout_transition=2",
                "-c:v", "copy",  # Copy video without re-encoding
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-y", str(output_path)
            ])
        elif temp_audio_paths[0][0] == "voice":
            # Voice only
            logger.info("🎤 Adding voice narration")
            cmd.extend([
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-y", str(output_path)
            ])
        else:
            # Music only
            logger.info("🎵 Adding background music")
            cmd.extend([
                "-filter_complex", "[1:a]volume=0.3[music]",  # Music at 30% volume
                "-map", "0:v",
                "-map", "[music]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-y", str(output_path)
            ])

        # Execute FFmpeg
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ Audio mixed successfully: {output_path}")
            return str(output_path)
        else:
            logger.error(f"❌ FFmpeg failed: {result.stderr}")
            raise Exception(f"Audio mixing failed: {result.stderr}")

    def create_reel(
        self,
        video_urls: List[str],
        music_url: Optional[str] = None,
        voice_url: Optional[str] = None,
        segment_duration: float = 5.0,
        output_filename: str = None
    ) -> str:
        """
        Complete reel creation pipeline

        Args:
            video_urls: List of Pexels video URLs
            music_url: Background music URL
            voice_url: Voice narration URL
            segment_duration: Duration of each video segment
            output_filename: Custom output filename

        Returns:
            Path to final reel
        """
        if not output_filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"reel_{timestamp}.mp4"

        logger.info("🎬 Starting reel creation pipeline...")
        logger.info(f"   Videos: {len(video_urls)}")
        logger.info(f"   Music: {'Yes' if music_url else 'No'}")
        logger.info(f"   Voice: {'Yes' if voice_url else 'No'}")

        try:
            # Step 1: Download videos
            video_paths = self.download_videos(video_urls)

            if not video_paths:
                raise Exception("No videos downloaded")

            # Step 2: Trim and concatenate
            concat_video = self.trim_and_concat_videos(
                video_paths,
                segment_duration=segment_duration
            )

            # Step 3: Mix audio
            final_video = self.mix_audio_with_video(
                concat_video,
                music_url=music_url,
                voice_url=voice_url,
                output_filename=output_filename
            )

            logger.info(f"🎉 Reel creation complete: {final_video}")
            return final_video

        except Exception as e:
            logger.error(f"❌ Reel creation failed: {e}")
            raise

        finally:
            # Cleanup temp files
            self.cleanup()

    def cleanup(self):
        """Remove temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")

    # ========== INTERNAL HELPER METHODS ==========

    def _download_audio(self, url: str, output_path: str):
        """Download audio file from URL"""
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"✅ Downloaded audio: {Path(output_path).name}")

    def _trim_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        output_path: str
    ) -> bool:
        """Trim video segment using FFmpeg and convert to 9:16 (720x1280)"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # FFmpeg filter to convert ANY aspect ratio to 9:16 (720x1280)
            # Strategy: Scale to fill 1280 height, then crop center to 720 width
            vf_filter = "scale=-2:1280,crop=720:1280:(in_w-720)/2:0"

            logger.info(f"📐 Converting to 9:16 (720x1280) portrait format")

            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-vf", vf_filter,  # Apply 9:16 conversion filter
                "-c:v", "libx264",
                "-preset", "faster",
                "-crf", "18",
                "-pix_fmt", "yuv420p",  # Ensure compatibility
                "-an",  # Remove audio (we'll add it later)
                "-y", output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"❌ FFmpeg trim failed for {video_path}")
                logger.error(f"STDERR: {result.stderr[:300]}")

            return result.returncode == 0

        except Exception as e:
            logger.error(f"❌ Exception trimming {video_path}: {e}")
            return False

    def _concat_clips(self, clip_paths: List[str], output_path: str) -> bool:
        """Concatenate video clips using FFmpeg concat demuxer (more reliable)"""
        try:
            if len(clip_paths) < 2:
                logger.error("Need at least 2 clips")
                return False

            # Verify all input files exist
            for clip in clip_paths:
                if not os.path.exists(clip):
                    logger.error(f"❌ Input file does not exist: {clip}")
                    return False

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create concat file list (more reliable than filter_complex)
            concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
            with open(concat_file, 'w') as f:
                for clip_path in clip_paths:
                    # Use absolute paths and escape single quotes
                    abs_path = os.path.abspath(clip_path)
                    f.write(f"file '{abs_path}'\n")

            logger.info(f"Created concat list: {concat_file}")

            # Use concat demuxer (simpler and more reliable)
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-y", output_path
            ]

            logger.info(f"Running FFmpeg concat (demuxer method)...")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up concat file
            try:
                os.remove(concat_file)
            except:
                pass

            if result.returncode == 0:
                logger.info(f"✅ Successfully concatenated {len(clip_paths)} clips")
                return True
            else:
                logger.error(f"❌ FFmpeg concat failed with return code {result.returncode}")
                logger.error(f"STDERR (full): {result.stderr}")
                logger.error(f"STDOUT (full): {result.stdout}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception during concatenation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


# Test
if __name__ == "__main__":
    mixer = VideoMixer()
    print("Video Mixer ready!")
