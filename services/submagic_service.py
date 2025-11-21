"""
Submagic Service - Auto-generate subtitles and apply effects
"""

import requests
import time
import logging
from typing import Optional, Dict
from config import Config

logger = logging.getLogger(__name__)


class SubmagicService:
    """
    Submagic API client for automated video editing.

    Features:
    - Auto-generated subtitles from audio
    - Professional caption styling
    - Transitions and effects
    - Social media optimization
    """

    def __init__(self):
        """Initialize Submagic service with API credentials"""
        self.api_key = Config.SUBMAGIC_API_KEY
        self.base_url = Config.SUBMAGIC_BASE_URL

        if not self.api_key:
            raise ValueError("SUBMAGIC_API_KEY not found in environment variables")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def submit_video(
        self,
        video_url: str,
        enable_subtitles: bool = True,
        caption_style: Optional[str] = None,
        add_transitions: Optional[bool] = None,
        aspect_ratio: Optional[str] = None
    ) -> Dict:
        """
        Submit a video to Submagic for editing.

        Args:
            video_url: Public URL of the video to edit
            enable_subtitles: Whether to add auto-generated captions
            caption_style: Caption style preset (default from Config)
            add_transitions: Add transitions between clips (default from Config)
            aspect_ratio: Video aspect ratio (default from Config)

        Returns:
            dict: Response with job_id for polling
            {
                "job_id": "abc123",
                "status": "processing",
                "message": "Video submitted successfully"
            }
        """
        url = f"{self.base_url}/videos/create"

        # Use config defaults if not specified
        caption_style = caption_style or Config.SUBMAGIC_CAPTION_STYLE
        add_transitions = add_transitions if add_transitions is not None else Config.SUBMAGIC_ADD_TRANSITIONS
        aspect_ratio = aspect_ratio or Config.SUBMAGIC_ASPECT_RATIO

        payload = {
            "video_url": video_url,
            "settings": {
                "add_captions": enable_subtitles,
                "caption_style": caption_style,
                "add_transitions": add_transitions,
                "add_music": Config.SUBMAGIC_ADD_MUSIC,  # We handle music separately
                "aspect_ratio": aspect_ratio
            }
        }

        logger.info(f"Submitting video to Submagic: {video_url[:60]}...")
        logger.info(f"Settings: subtitles={enable_subtitles}, style={caption_style}")

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Video submitted successfully. Job ID: {result.get('job_id')}")

            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Submagic API error: {e}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise Exception(f"Failed to submit video to Submagic: {e}")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {e}")
            raise Exception(f"Network error while submitting to Submagic: {e}")

    def get_job_status(self, job_id: str) -> Dict:
        """
        Check the status of a Submagic editing job.

        Args:
            job_id: The job ID returned from submit_video()

        Returns:
            dict: Job status information
            {
                "job_id": "abc123",
                "status": "completed" | "processing" | "failed",
                "progress": 75,  # percentage
                "download_url": "https://...",  # only if completed
                "error": "error message"  # only if failed
            }
        """
        url = f"{self.base_url}/videos/{job_id}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get job status: {e}")
            raise Exception(f"Failed to get Submagic job status: {e}")

    def wait_for_completion(
        self,
        job_id: str,
        timeout: int = None,
        poll_interval: int = 15
    ) -> str:
        """
        Wait for Submagic editing to complete.

        Args:
            job_id: The job ID to poll
            timeout: Maximum time to wait in seconds (default from Config)
            poll_interval: Time between status checks in seconds

        Returns:
            str: Download URL of the edited video

        Raises:
            TimeoutError: If job doesn't complete within timeout
            Exception: If job fails
        """
        timeout = timeout or Config.SUBMAGIC_TIMEOUT
        start_time = time.time()

        logger.info(f"⏳ Waiting for Submagic job {job_id} to complete...")
        logger.info(f"   Timeout: {timeout}s | Poll interval: {poll_interval}s")

        while time.time() - start_time < timeout:
            try:
                status = self.get_job_status(job_id)

                job_status = status.get("status", "unknown")
                progress = status.get("progress", 0)

                if job_status == "completed":
                    download_url = status.get("download_url")
                    if not download_url:
                        raise Exception("Job completed but no download URL provided")

                    logger.info(f"✅ Submagic editing complete!")
                    return download_url

                elif job_status == "failed":
                    error_msg = status.get("error", "Unknown error")
                    logger.error(f"❌ Submagic job failed: {error_msg}")
                    raise Exception(f"Submagic editing failed: {error_msg}")

                elif job_status == "processing":
                    elapsed = int(time.time() - start_time)
                    logger.info(f"   Progress: {progress}% | Elapsed: {elapsed}s")
                    time.sleep(poll_interval)

                else:
                    logger.warning(f"⚠️  Unknown status: {job_status}")
                    time.sleep(poll_interval)

            except Exception as e:
                if "failed" in str(e).lower():
                    raise  # Re-raise job failures
                logger.warning(f"⚠️  Error checking status: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        elapsed = int(time.time() - start_time)
        raise TimeoutError(f"Submagic editing timed out after {elapsed}s (limit: {timeout}s)")

    def download_video(self, download_url: str, output_path: str) -> str:
        """
        Download the edited video from Submagic.

        Args:
            download_url: The download URL from Submagic
            output_path: Local path to save the video

        Returns:
            str: Path to the downloaded file
        """
        logger.info(f"📥 Downloading edited video from Submagic...")

        try:
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 1MB
                        if downloaded % (1024 * 1024) == 0:
                            mb_downloaded = downloaded / (1024 * 1024)
                            logger.info(f"   Downloaded: {mb_downloaded:.1f} MB")

            file_size_mb = downloaded / (1024 * 1024)
            logger.info(f"✅ Video downloaded: {output_path}")
            logger.info(f"   Size: {file_size_mb:.2f} MB")

            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download video: {e}")
            raise Exception(f"Failed to download edited video: {e}")


# Test the service
if __name__ == "__main__":
    print("🧪 Testing Submagic Service\n")

    # This is a test URL - replace with actual video URL
    test_video_url = "https://example.com/test_video.mp4"

    try:
        service = SubmagicService()
        print("✓ Service initialized")

        # Submit video
        result = service.submit_video(test_video_url, enable_subtitles=True)
        job_id = result.get("job_id")
        print(f"✓ Video submitted: {job_id}")

        # Wait for completion (this will take a while)
        download_url = service.wait_for_completion(job_id)
        print(f"✓ Editing complete: {download_url}")

        # Download
        service.download_video(download_url, "test_output.mp4")
        print("✓ Video downloaded")

    except Exception as e:
        print(f"❌ Test failed: {e}")
