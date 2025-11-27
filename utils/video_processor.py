"""
Video Processor Utility
FFmpeg wrapper for video processing operations
"""

import os
import subprocess
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from config import Config


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing using FFmpeg"""

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize video processor

        Args:
            temp_dir: Temporary directory for intermediate files (optional)
        """
        import tempfile
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="video_processor_"))
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Video processor temp directory: {self.temp_dir}")

        # Check if FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video metadata using ffprobe

        Args:
            video_path: Path to video file

        Returns:
            Dict with video metadata (duration, width, height, fps)
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
                "-of", "json",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)

            stream = data.get("streams", [{}])[0]
            format_info = data.get("format", {})

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

            # Get duration
            duration = float(stream.get("duration", format_info.get("duration", 0)))

            info = {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": duration
            }

            logger.info(f"Video info for {Path(video_path).name}: {info['width']}x{info['height']}, "
                       f"{info['duration']:.2f}s, {info['fps']:.2f} fps")

            return info

        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            raise

    def trim_video(
        self,
        input_path: str,
        output_path: str,
        start_time: float = 0,
        duration: Optional[float] = None
    ) -> str:
        """
        Trim video to specified duration

        Args:
            input_path: Path to input video
            output_path: Path to save trimmed video
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (if None, keeps full video)

        Returns:
            Path to trimmed video
        """
        print(f"\n✂️  Trimming video...")
        print(f"   Input: {Path(input_path).name}")
        print(f"   Start: {start_time}s, Duration: {duration}s")

        try:
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", input_path
            ]

            if duration:
                cmd.extend(["-t", str(duration)])

            cmd.extend([
                "-c:v", Config.VIDEO_CODEC,
                "-c:a", Config.AUDIO_CODEC,
                "-y", output_path
            ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Trimmed video saved: {Path(output_path).name}")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Video trim failed: {str(e)}")
            raise

    def calculate_clip_duration(self, num_clips: int, target_total: int = 20) -> float:
        """
        Calculate duration per clip to reach target total

        Args:
            num_clips: Number of clips
            target_total: Target total duration in seconds (default: 20)

        Returns:
            Duration per clip in seconds
        """
        if num_clips <= 0:
            raise ValueError("Number of clips must be positive")

        duration = target_total / num_clips
        print(f"\n📊 Clip Duration Calculation:")
        print(f"   {num_clips} clips → {duration:.2f}s per clip = {target_total}s total")

        return duration

    def concatenate_videos(self, video_paths: List[str], output_path: str) -> str:
        """
        Concatenate multiple videos into one

        Args:
            video_paths: List of video file paths
            output_path: Path to save concatenated video

        Returns:
            Path to concatenated video
        """
        if not video_paths:
            raise ValueError("No video paths provided")

        print(f"\n🔗 Concatenating {len(video_paths)} videos...")

        try:
            # Create concat file
            concat_file = self.temp_dir / "concat_list.txt"

            with open(concat_file, 'w') as f:
                for path in video_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")

            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-y", output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Videos concatenated: {Path(output_path).name}")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Video concatenation failed: {str(e)}")
            raise

    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_volume: float = 1.0
    ) -> str:
        """
        Add audio track to video

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to save output
            audio_volume: Audio volume multiplier (default: 1.0)

        Returns:
            Path to output video with audio
        """
        print(f"\n🎵 Adding audio to video...")
        print(f"   Video: {Path(video_path).name}")
        print(f"   Audio: {Path(audio_path).name}")
        print(f"   Volume: {audio_volume}")

        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex", f"[1:a]volume={audio_volume}[a]",
                "-map", "0:v",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", Config.AUDIO_CODEC,
                "-shortest",
                "-y", output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Audio added: {Path(output_path).name}")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Add audio failed: {str(e)}")
            raise

    def mix_audio_tracks(
        self,
        video_path: str,
        music_path: Optional[str],
        voice_path: Optional[str],
        output_path: str,
        music_volume: float = 0.15,
        voice_volume: float = 1.0
    ) -> str:
        """
        Mix multiple audio tracks with video

        Args:
            video_path: Path to video file
            music_path: Path to music file (optional)
            voice_path: Path to voice file (optional)
            output_path: Path to save output
            music_volume: Music volume multiplier (default: 0.15 = 15%)
            voice_volume: Voice volume multiplier (default: 1.0 = 100%)

        Returns:
            Path to output video with mixed audio
        """
        print(f"\n🎵 Mixing audio tracks...")

        if not music_path and not voice_path:
            logger.warning("No audio provided, copying video as-is")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path

        try:
            # Get video duration
            video_info = self.get_video_info(video_path)
            video_duration = video_info["duration"]

            cmd = ["ffmpeg", "-i", video_path]

            if voice_path and music_path:
                # Both voice and music
                print(f"   Mixing: Voice (100%) + Music ({int(music_volume*100)}%)")

                cmd.extend(["-i", voice_path, "-i", music_path])
                cmd.extend([
                    "-filter_complex",
                    f"[1:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,volume={voice_volume}[voice];"
                    f"[2:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,volume={music_volume}[music];"
                    f"[voice][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", Config.AUDIO_CODEC,
                    "-b:a", Config.AUDIO_BITRATE,
                    "-t", str(video_duration),
                    "-y", output_path
                ])

            elif voice_path:
                # Voice only
                print(f"   Adding: Voice narration only")
                cmd.extend(["-i", voice_path])
                cmd.extend([
                    "-c:v", "copy",
                    "-c:a", Config.AUDIO_CODEC,
                    "-shortest",
                    "-y", output_path
                ])

            elif music_path:
                # Music only
                print(f"   Adding: Background music ({int(music_volume*100)}%)")
                cmd.extend(["-i", music_path])
                cmd.extend([
                    "-filter_complex", f"[1:a]volume={music_volume}[music]",
                    "-map", "0:v",
                    "-map", "[music]",
                    "-c:v", "copy",
                    "-c:a", Config.AUDIO_CODEC,
                    "-shortest",
                    "-y", output_path
                ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Audio mixed: {Path(output_path).name}")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Audio mixing failed: {str(e)}")
            raise

    def composite_avatar_with_broll(
        self,
        broll_path: str,
        avatar_path: str,
        output_path: str,
        avatar_scale: float = 1.0,
        position: str = "bottom-center",
        padding: int = 0
    ) -> str:
        """
        Composite avatar video over B-roll (Instagram Reels style)

        Args:
            broll_path: Path to B-roll background video
            avatar_path: Path to avatar video (with transparent background)
            output_path: Path to save composited video
            avatar_scale: Avatar height as fraction of video (default: 1.0 = full height, we'll crop bottom half)
            position: Avatar position - "bottom-center" for Instagram style
            padding: Padding from edges in pixels (default: 0)

        Returns:
            Path to composited video
        """
        print(f"\n🎬 Compositing avatar over B-roll (Instagram Reels style)...")
        print(f"   B-roll: {Path(broll_path).name}")
        print(f"   Avatar: {Path(avatar_path).name}")
        print(f"   Position: {position} (bottom half, centered)")

        try:
            # Get video dimensions
            broll_info = self.get_video_info(broll_path)
            broll_width = broll_info["width"]
            broll_height = broll_info["height"]

            # Avatar will be FULL WIDTH, positioned at BOTTOM HALF
            avatar_width = broll_width
            avatar_height = broll_height  # Keep original height, we'll position it

            # Position: bottom-center (centered horizontally, bottom half vertically)
            x = "(W-w)/2"  # Centered horizontally
            y = "H/2"      # Start at 50% from top (bottom half)

            # FFmpeg filter complex:
            # SIMPLE APPROACH: Scale avatar to bottom half, overlay directly
            # HeyGen's matting=true already removes background
            avatar_height_scaled = broll_height // 2  # 50% height for bottom half

            cmd = [
                "ffmpeg",
                "-i", broll_path,
                "-i", avatar_path,
                "-filter_complex",
                # Scale avatar to full width and 50% height (bottom half)
                f"[1:v]scale={avatar_width}:{avatar_height_scaled}[avatar];"
                f"[0:v][avatar]overlay={x}:{y}:shortest=1[v]",
                "-map", "[v]",
                "-map", "1:a?",  # Use avatar's audio if available
                "-c:v", Config.VIDEO_CODEC,
                "-c:a", Config.AUDIO_CODEC,
                "-shortest",
                "-y", output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Avatar composited: {Path(output_path).name}")
                print(f"   📐 Layout: B-roll (full) + Avatar (bottom half, waist-up)")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Avatar compositing failed: {str(e)}")
            raise

    def resize_to_9_16(self, input_path: str, output_path: str) -> str:
        """
        Resize video to 9:16 aspect ratio (720x1280)

        Args:
            input_path: Path to input video
            output_path: Path to save resized video

        Returns:
            Path to resized video
        """
        print(f"\n📐 Resizing to 9:16 (720x1280)...")

        try:
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"scale={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
                       f"pad={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", Config.VIDEO_CODEC,
                "-c:a", "copy",
                "-y", output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Video resized: {Path(output_path).name}")
                return output_path
            else:
                raise Exception(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            logger.error(f"Video resize failed: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")


# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()

    # Test video info
    info = processor.get_video_info("test_video.mp4")
    print(f"Video Info: {info}")

    # Test trim
    processor.trim_video("test_video.mp4", "trimmed.mp4", duration=10)

    # Test concatenate
    processor.concatenate_videos(["video1.mp4", "video2.mp4"], "concat.mp4")

    # Clean up
    processor.cleanup()
