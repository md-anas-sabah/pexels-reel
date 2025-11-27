"""
HeyGen API Service
Handles avatar video generation with background compositing
"""

import os
import time
import requests
from typing import Dict, Optional
from config import Config


class HeyGenService:
    """Service for interacting with HeyGen API to generate avatar videos"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize HeyGen service

        Args:
            api_key: HeyGen API key (defaults to Config.HEYGEN_API_KEY)
        """
        self.api_key = api_key or Config.HEYGEN_API_KEY
        self.base_url = Config.HEYGEN_BASE_URL

        if not self.api_key:
            raise ValueError("HeyGen API key is required")

        self.headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def generate_video(
        self,
        script: str,
        avatar_id: str,
        voice_id: str,
        background_type: str = None,
        background_value: str = None,
        background_url: Optional[str] = None,
        title: Optional[str] = None,
        emotion: str = "Excited"
    ) -> Dict:
        """
        Generate avatar video with HeyGen API

        Args:
            script: Text for avatar to speak
            avatar_id: HeyGen avatar ID
            voice_id: HeyGen voice ID for speech
            background_type: "color", "video", or None (default: None = avatar's original background)
            background_value: Color hex code (if type is "color")
            background_url: Video URL (if type is "video")
            title: Optional video title
            emotion: Voice emotion (default: "Excited")
                     Can be either HeyGen format or AI Agent format
                     Will be auto-mapped to HeyGen-compatible emotion

        Returns:
            Dict with video_id and other response data
        """
        # Map emotion to HeyGen-compatible format if needed
        # HeyGen only accepts: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster', 'Angry'
        emotion_lower = emotion.lower()
        if emotion_lower in Config.HEYGEN_EMOTION_MAPPING:
            mapped_emotion = Config.HEYGEN_EMOTION_MAPPING[emotion_lower]
            print(f"   🎭 Emotion mapping: '{emotion}' → '{mapped_emotion}'")
            emotion = mapped_emotion
        elif emotion not in ['Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster', 'Angry']:
            # Fallback to Friendly if unknown emotion
            print(f"   ⚠️  Unknown emotion '{emotion}', using 'Friendly'")
            emotion = "Friendly"

        # Build payload based on HeyGen API format
        payload = {
            "caption": False,
            "title": title or "AI Generated Reel",
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": avatar_id,
                        "scale": 1.8,  # Zoom in to fill full screen (1.8x zoom)
                        "avatar_style": "normal",
                        "offset": {
                            "x": 0,      # Centered horizontally
                            "y": 0       # Centered vertically (0 = perfect center, shows full avatar from top to bottom)
                        },
                        "talking_style": "stable",
                        "expression": "default",
                        "super_resolution": True,
                        "matting": False  # No matting - using solid background
                    },
                    "voice": {
                        "type": "text",
                        "voice_id": voice_id,
                        "input_text": script,
                        "speed": 1,
                        "pitch": 0,
                        "emotion": emotion
                    }
                }
            ],
            "dimension": {
                "width": 720,    # SD resolution (9:16 aspect ratio for Reels)
                "height": 1280   # Supported by Basic/Free HeyGen plan
                                 # TODO: Update to 1080x1920 after upgrading to Creator plan
            }
        }

        # Only add background if specified (otherwise uses avatar's original background)
        if background_type is not None:
            payload["video_inputs"][0]["background"] = {
                "type": background_type,
                "value": background_value if background_type == "color" else background_url
            }

        # If background is video, update the payload with required play_style
        # HeyGen API requires play_style for video backgrounds
        # Options: "fit_to_scene", "freeze", "loop", "full_video"
        # Using "fit_to_scene" to stretch B-roll to match avatar duration
        if background_type == "video" and background_url:
            payload["video_inputs"][0]["background"] = {
                "type": "video",
                "url": background_url,
                "play_style": "fit_to_scene"  # Stretch B-roll to match avatar speech duration
            }

        print(f"\n🎬 Generating HeyGen Avatar Video...")
        print(f"   Avatar: {avatar_id}")
        print(f"   Voice: {voice_id}")
        print(f"   Script Length: {len(script)} characters")
        print(f"   Background: {'Avatar Original' if background_type is None else background_type}")

        try:
            # Submit video generation request
            response = requests.post(
                f"{self.base_url}/video/generate",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                raise Exception(f"HeyGen API Error: {result.get('error')}")

            video_id = result.get("data", {}).get("video_id")

            if not video_id:
                raise Exception("No video_id returned from HeyGen API")

            print(f"   ✅ Video generation started! Video ID: {video_id}")

            return {
                "video_id": video_id,
                "status": "processing",
                "data": result.get("data", {})
            }

        except requests.exceptions.RequestException as e:
            print(f"   ❌ HeyGen API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise

    def get_video_status(self, video_id: str) -> Dict:
        """
        Check the status of a HeyGen video generation

        Args:
            video_id: HeyGen video ID

        Returns:
            Dict with status and video_url (if completed)
        """
        try:
            # Use v1 endpoint for video status (not v2)
            # Endpoint: GET https://api.heygen.com/v1/video_status.get?video_id={video_id}
            response = requests.get(
                "https://api.heygen.com/v1/video_status.get",
                headers=self.headers,
                params={"video_id": video_id},
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            status = result.get("data", {}).get("status", "unknown")
            video_url = result.get("data", {}).get("video_url")

            return {
                "status": status,
                "video_url": video_url,
                "data": result.get("data", {})
            }

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Failed to get video status: {str(e)}")
            raise

    def wait_for_completion(
        self,
        video_id: str,
        timeout: int = None,
        poll_interval: int = 10
    ) -> str:
        """
        Poll HeyGen API until video is ready (NO TIMEOUT - waits indefinitely)

        Args:
            video_id: HeyGen video ID
            timeout: IGNORED - waits indefinitely until video completes
            poll_interval: Seconds between status checks (default: 10)

        Returns:
            Video URL when ready

        Raises:
            Exception: If video generation fails
        """
        start_time = time.time()

        print(f"\n⏳ Waiting for HeyGen video completion...")
        print(f"   Video ID: {video_id}")
        print(f"   ⚠️  NO TIMEOUT - Will wait indefinitely until video completes")

        while True:
            elapsed = time.time() - start_time

            # NO TIMEOUT CHECK - removed completely

            try:
                status_data = self.get_video_status(video_id)
                status = status_data.get("status", "").lower()

                print(f"   Status: {status} (elapsed: {int(elapsed)}s)")

                if status == "completed":
                    video_url = status_data.get("video_url")

                    if not video_url:
                        raise Exception("Video completed but no URL returned")

                    print(f"   ✅ Video ready! URL: {video_url}")
                    return video_url

                elif status == "failed" or status == "error":
                    error_msg = status_data.get("data", {}).get("error", "Unknown error")
                    raise Exception(f"HeyGen video generation failed: {error_msg}")

                elif status in ["processing", "pending", "queued"]:
                    # Still processing, continue waiting
                    time.sleep(poll_interval)
                    continue

                else:
                    print(f"   ⚠️  Unknown status: {status}, continuing to wait...")
                    time.sleep(poll_interval)

            except requests.exceptions.RequestException as e:
                print(f"   ⚠️  Network error while checking status: {str(e)}")
                print(f"   Retrying in {poll_interval}s...")
                time.sleep(poll_interval)

    def download_video(self, video_url: str, output_path: str) -> str:
        """
        Download HeyGen video from URL

        Args:
            video_url: URL of the generated video
            output_path: Local path to save video

        Returns:
            Path to downloaded video
        """
        print(f"\n📥 Downloading HeyGen video...")
        print(f"   URL: {video_url}")
        print(f"   Output: {output_path}")

        try:
            response = requests.get(video_url, stream=True, timeout=120)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"   Progress: {progress:.1f}%", end='\r')

            print(f"\n   ✅ Download complete! Saved to: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Download failed: {str(e)}")
            raise

    def generate_video_with_broll(
        self,
        script: str,
        avatar_id: str,
        voice_id: str,
        broll_url: str,
        title: Optional[str] = None,
        emotion: str = "Excited"
    ) -> Dict:
        """
        Convenience method: Generate avatar video with B-roll background

        Args:
            script: Text for avatar to speak
            avatar_id: HeyGen avatar ID
            voice_id: HeyGen voice ID
            broll_url: Public URL to B-roll video (Supabase URL)
            title: Optional video title
            emotion: Voice emotion

        Returns:
            Dict with video_id and response data
        """
        return self.generate_video(
            script=script,
            avatar_id=avatar_id,
            voice_id=voice_id,
            background_type="video",
            background_url=broll_url,
            title=title,
            emotion=emotion
        )


# Example usage
if __name__ == "__main__":
    # Test HeyGen service
    service = HeyGenService()

    # Example: Generate video with color background
    result = service.generate_video(
        script="Hello! This is a test video generated with HeyGen AI.",
        avatar_id="7e0f7c966e514ee69e8592550886169a",  # Boy avatar
        voice_id="your_voice_id_here",
        background_type="color",
        background_value="#000000"
    )

    print(f"Video ID: {result['video_id']}")

    # Wait for completion
    video_url = service.wait_for_completion(result['video_id'])

    # Download video
    service.download_video(video_url, "output/test_heygen_video.mp4")
