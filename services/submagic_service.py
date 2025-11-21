"""
Submagic Service - Auto-generate subtitles and apply effects
Official API Documentation: https://docs.submagic.co/
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
    - Auto-generated subtitles from audio (98%+ accuracy)
    - Professional caption styling with 100+ templates
    - Transitions and effects
    - Social media optimization (9:16 aspect ratio)
    - 100+ language support

    Official API Workflow:
    1. Create Project (POST /v1/projects) -> returns project_id
    2. Poll Status (GET /v1/projects/{id}) -> wait for processing
    3. Export Video (POST /v1/projects/{id}/export) -> get download URL
    4. Download final video
    """

    def __init__(self):
        """Initialize Submagic service with API credentials"""
        self.api_key = Config.SUBMAGIC_API_KEY
        self.base_url = Config.SUBMAGIC_BASE_URL

        if not self.api_key:
            raise ValueError("SUBMAGIC_API_KEY not found in environment variables")

        # Submagic uses x-api-key header (NOT Authorization Bearer)
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def create_project(
        self,
        video_url: str,
        title: str = "AI Generated Reel",
        language: str = "en",
        template_name: str = "Iman",
        magic_zooms: bool = True,
        magic_brolls: bool = True,
        magic_brolls_percentage: int = 60,
        webhook_url: Optional[str] = None
    ) -> Dict:
        """
        Create a new Submagic project (Step 1 of workflow).

        This initiates video download, transcription, and AI processing.

        Args:
            video_url: Public URL of the video to edit (must be accessible)
            title: Project title for organization
            language: Language code (en, hi, ar, etc.) - 100+ supported
            template_name: Caption template (e.g., "Iman", "Hormozi 2", "MrBeast", "Alex")
            magic_zooms: Enable AI auto-zoom effects (default: True)
            magic_brolls: Enable AI B-roll insertion (default: True)
            magic_brolls_percentage: B-roll density 0-100 (default: 60)
            webhook_url: Optional webhook URL for completion notification

        Returns:
            dict: Project response with project ID
            {
                "id": "proj_abc123",
                "title": "AI Generated Reel",
                "status": "processing",
                "createdAt": "2025-01-21T10:30:00Z"
            }
        """
        url = f"{self.base_url}/projects"

        # Submagic API request body structure (matching n8n working payload)
        payload = {
            "title": title,
            "videoUrl": video_url,
            "language": language,
            "templateName": template_name,
            "magicZooms": magic_zooms,
            "magicBrolls": magic_brolls,
            "magicBrollsPercentage": magic_brolls_percentage
        }

        # Add webhook URL if provided
        if webhook_url:
            payload["webhookUrl"] = webhook_url

        logger.info(f"📤 Creating Submagic project...")
        logger.info(f"   Video URL: {video_url[:70]}...")
        logger.info(f"   Language: {language} | Template: {template_name}")
        logger.info(f"   Magic Zooms: {magic_zooms} | Magic B-rolls: {magic_brolls} ({magic_brolls_percentage}%)")

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            project_id = result.get('id')

            logger.info(f"✅ Project created successfully!")
            logger.info(f"   Project ID: {project_id}")

            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Submagic API error: {e}")
            if hasattr(e, 'response'):
                logger.error(f"   Status Code: {e.response.status_code}")
                logger.error(f"   Response: {e.response.text}")
            raise Exception(f"Failed to create Submagic project: {e}")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {e}")
            raise Exception(f"Network error while submitting to Submagic: {e}")

    def get_project_status(self, project_id: str) -> Dict:
        """
        Check the status of a Submagic project (Step 2 of workflow).

        Args:
            project_id: The project ID returned from create_project()

        Returns:
            dict: Project status information
            {
                "id": "proj_abc123",
                "status": "ready" | "processing" | "failed",
                "title": "AI Generated Reel",
                "outputUrl": "https://...",  # if processing complete
                "error": "error message"  # only if failed
            }
        """
        url = f"{self.base_url}/projects/{project_id}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get project status: {e}")
            raise Exception(f"Failed to get Submagic project status: {e}")

    def export_project(self, project_id: str) -> Dict:
        """
        Export the processed video (Step 3 of workflow).

        This triggers final rendering and provides the download URL.

        Args:
            project_id: The project ID to export

        Returns:
            dict: Export information with download URL
            {
                "id": "proj_abc123",
                "exportUrl": "https://cdn.submagic.co/...",
                "status": "completed"
            }
        """
        url = f"{self.base_url}/projects/{project_id}/export"

        logger.info(f"📤 Exporting project {project_id}...")

        try:
            response = requests.post(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            result = response.json()

            # DEBUG: Log the full response to understand the structure
            logger.info(f"📋 Export API Response: {result}")

            # Try multiple possible field names for the download URL
            export_url = (result.get('exportUrl') or
                         result.get('downloadUrl') or
                         result.get('url') or
                         result.get('video_url') or
                         result.get('output_url'))

            if export_url:
                logger.info(f"✅ Export complete!")
                logger.info(f"   Download URL: {export_url[:80]}...")
            else:
                logger.warning(f"⚠️  Export succeeded but no URL found in response")
                logger.warning(f"   Available fields: {list(result.keys())}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to export project: {e}")
            if hasattr(e, 'response'):
                logger.error(f"   Response: {e.response.text}")
            raise Exception(f"Failed to export Submagic project: {e}")

    def wait_for_completion(
        self,
        project_id: str,
        timeout: int = None,
        poll_interval: int = 15
    ) -> Dict:
        """
        Wait for Submagic project processing to complete (Step 2 continued).

        Polls the project status until it's ready for export.

        Args:
            project_id: The project ID to poll
            timeout: Maximum time to wait in seconds (default from Config)
            poll_interval: Time between status checks in seconds

        Returns:
            dict: Project data when ready
            {
                "id": "proj_abc123",
                "status": "ready",
                "outputUrl": "https://..."
            }

        Raises:
            TimeoutError: If project doesn't complete within timeout
            Exception: If project fails
        """
        timeout = timeout or Config.SUBMAGIC_TIMEOUT
        start_time = time.time()

        logger.info(f"⏳ Waiting for Submagic project {project_id} to be ready...")
        logger.info(f"   Timeout: {timeout}s | Poll interval: {poll_interval}s")

        while time.time() - start_time < timeout:
            try:
                status = self.get_project_status(project_id)

                project_status = status.get("status", "unknown")
                elapsed = int(time.time() - start_time)

                # Accept both "ready" and "completed" as success states
                if project_status in ["ready", "completed"]:
                    logger.info(f"✅ Submagic project ready for export!")
                    logger.info(f"   Total processing time: {elapsed}s")
                    logger.info(f"📋 Project Status Fields: {list(status.keys())}")
                    return status

                elif project_status == "failed":
                    error_msg = status.get("error", "Unknown error")
                    logger.error(f"❌ Submagic project failed: {error_msg}")
                    raise Exception(f"Submagic processing failed: {error_msg}")

                elif project_status == "processing":
                    logger.info(f"   Status: Processing | Elapsed: {elapsed}s")
                    time.sleep(poll_interval)

                else:
                    logger.info(f"   Status: {project_status} | Elapsed: {elapsed}s")
                    time.sleep(poll_interval)

            except Exception as e:
                if "failed" in str(e).lower():
                    raise  # Re-raise project failures
                logger.warning(f"⚠️  Error checking status: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        elapsed = int(time.time() - start_time)
        raise TimeoutError(f"Submagic processing timed out after {elapsed}s (limit: {timeout}s)")

    def download_video(self, download_url: str, output_path: str, max_retries: int = 3) -> str:
        """
        Download the edited video from Submagic (Step 4 of workflow).

        Args:
            download_url: The download URL from export_project()
            output_path: Local path to save the video
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            str: Path to the downloaded file
        """
        import time

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📥 Downloading edited video from Submagic... (Attempt {attempt}/{max_retries})")

                # Increased timeout to 180 seconds (3 minutes) for large files
                response = requests.get(download_url, stream=True, timeout=180)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                last_log_mb = 0

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 5MB
                            mb_downloaded = downloaded / (1024 * 1024)
                            if mb_downloaded - last_log_mb >= 5:
                                percent = (downloaded / total_size * 100) if total_size > 0 else 0
                                logger.info(f"   Downloaded: {mb_downloaded:.1f} MB ({percent:.1f}%)")
                                last_log_mb = mb_downloaded

                file_size_mb = downloaded / (1024 * 1024)
                logger.info(f"✅ Video downloaded: {output_path}")
                logger.info(f"   Size: {file_size_mb:.2f} MB")

                return output_path

            except requests.exceptions.Timeout as e:
                logger.warning(f"⚠️  Download timeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    wait_time = attempt * 5  # Exponential backoff: 5s, 10s, 15s
                    logger.info(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to download after {max_retries} attempts")
                    raise Exception(f"Download timed out after {max_retries} attempts: {e}")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Failed to download video: {e}")
                if attempt < max_retries and "timeout" in str(e).lower():
                    wait_time = attempt * 5
                    logger.info(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to download edited video: {e}")

    def _download_with_wget(self, download_url: str, output_path: str) -> str:
        """
        Fallback download method using wget or curl (more reliable for large files)
        """
        import subprocess
        import shutil

        logger.info(f"📥 Using wget/curl for more reliable download...")

        # Try wget first
        if shutil.which('wget'):
            cmd = ['wget', '-O', output_path, '--timeout=300', download_url]
            logger.info(f"   Using wget...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
            if result.returncode == 0:
                logger.info(f"✅ Video downloaded with wget: {output_path}")
                return output_path

        # Try curl as fallback
        if shutil.which('curl'):
            cmd = ['curl', '-L', '-o', output_path, '--max-time', '300', download_url]
            logger.info(f"   Using curl...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
            if result.returncode == 0:
                logger.info(f"✅ Video downloaded with curl: {output_path}")
                return output_path

        raise Exception("wget/curl download also failed")

    def process_video(
        self,
        video_url: str,
        output_path: str,
        title: str = "AI Generated Reel",
        language: str = "en",
        template_name: str = "Iman",
        magic_zooms: bool = True,
        magic_brolls: bool = True,
        magic_brolls_percentage: int = 60
    ) -> str:
        """
        Complete end-to-end workflow: Create → Wait → Export → Download.

        This is the main method that combines all 4 steps:
        1. Create project
        2. Wait for processing
        3. Export video
        4. Download final result

        Args:
            video_url: Public URL of the video to edit
            output_path: Local path to save the final edited video
            title: Project title
            language: Language code (en, hi, ar, etc.)
            template_name: Caption template name
            magic_zooms: Enable AI auto-zoom effects
            magic_brolls: Enable AI B-roll insertion
            magic_brolls_percentage: B-roll density 0-100

        Returns:
            str: Path to the downloaded edited video

        Example:
            >>> service = SubmagicService()
            >>> final_video = service.process_video(
            ...     video_url="https://supabase.co/storage/my-video.mp4",
            ...     output_path="output/final_reel.mp4",
            ...     language="en",
            ...     template_name="Iman",
            ...     magic_zooms=True,
            ...     magic_brolls=True,
            ...     magic_brolls_percentage=60
            ... )
        """
        logger.info("🎬 Starting complete Submagic workflow...")

        # Step 1: Create project
        project = self.create_project(
            video_url=video_url,
            title=title,
            language=language,
            template_name=template_name,
            magic_zooms=magic_zooms,
            magic_brolls=magic_brolls,
            magic_brolls_percentage=magic_brolls_percentage
        )
        project_id = project.get('id')

        # Step 2: Wait for processing
        project_status = self.wait_for_completion(project_id)

        # Check if download URL is already in project status
        # Try downloadUrl first, then directUrl, then previewUrl as fallback
        download_url = (project_status.get('downloadUrl') or
                       project_status.get('directUrl') or
                       project_status.get('previewUrl') or
                       project_status.get('exportUrl') or
                       project_status.get('outputUrl') or
                       project_status.get('url') or
                       project_status.get('video_url'))

        # Store all available URLs for fallback
        download_url_primary = project_status.get('downloadUrl')
        download_url_direct = project_status.get('directUrl')
        download_url_preview = project_status.get('previewUrl')

        if download_url:
            logger.info(f"✅ Download URL found in project status")
            logger.info(f"   Primary URL: {'downloadUrl' if download_url_primary else 'directUrl'}")
            logger.info(f"   Skipping export step (URL already available)")
        else:
            # Step 3: Export video if URL not in status
            logger.info("📤 Download URL not in status, calling export endpoint...")
            export_result = self.export_project(project_id)
            download_url = (export_result.get('exportUrl') or
                           export_result.get('downloadUrl') or
                           export_result.get('url') or
                           export_result.get('video_url') or
                           export_result.get('output_url'))

        if not download_url:
            logger.error(f"❌ No download URL found")
            logger.error(f"   Project status fields: {list(project_status.keys())}")
            raise Exception("Export completed but no download URL provided in project status or export response")

        # Step 4: Download final video (with multiple fallback strategies)
        download_success = False

        # Try 1: Primary download URL (usually S3 accelerated)
        if download_url_primary:
            try:
                logger.info(f"📥 Trying primary download URL (downloadUrl)...")
                final_path = self.download_video(download_url_primary, output_path)
                download_success = True
            except Exception as e:
                logger.warning(f"⚠️  Primary download failed: {str(e)[:100]}")

        # Try 2: Direct URL (direct S3)
        if not download_success and download_url_direct:
            try:
                logger.warning(f"⚠️  Trying direct URL (directUrl)...")
                final_path = self.download_video(download_url_direct, output_path)
                download_success = True
            except Exception as e:
                logger.warning(f"⚠️  Direct download failed: {str(e)[:100]}")

        # Try 3: Preview URL
        if not download_success and download_url_preview:
            try:
                logger.warning(f"⚠️  Trying preview URL (previewUrl)...")
                final_path = self.download_video(download_url_preview, output_path)
                download_success = True
            except Exception as e:
                logger.warning(f"⚠️  Preview download failed: {str(e)[:100]}")

        # Try 4: Use wget/curl as fallback (more reliable for large files)
        if not download_success and download_url_direct:
            try:
                logger.warning(f"⚠️  Trying wget/curl fallback...")
                final_path = self._download_with_wget(download_url_direct, output_path)
                download_success = True
            except Exception as e:
                logger.warning(f"⚠️  wget/curl download failed: {str(e)[:100]}")

        # If all downloads failed, provide manual download instructions
        if not download_success:
            logger.error(f"❌ All download attempts failed!")
            logger.error(f"")
            logger.error(f"📋 MANUAL DOWNLOAD REQUIRED:")
            logger.error(f"   1. Open this URL in your browser:")
            if download_url_direct:
                logger.error(f"      {download_url_direct}")
            elif download_url_preview:
                logger.error(f"      {download_url_preview}")
            elif download_url_primary:
                logger.error(f"      {download_url_primary}")
            logger.error(f"")
            logger.error(f"   2. Save the video to: {output_path}")
            logger.error(f"")
            raise Exception(f"Automatic download failed. Please download manually from the URL above.")

        logger.info(f"🎉 Complete workflow finished successfully!")
        logger.info(f"   Final video saved: {final_path}")

        return final_path


# Test the service
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("🧪 Testing Submagic Service\n")
    print("=" * 80)

    # This is a test URL - replace with actual video URL from Supabase
    test_video_url = "https://pfixecrevydvermbgwmo.supabase.co/storage/v1/object/sign/reel-videos/test.mp4?token=..."

    try:
        service = SubmagicService()
        print("✅ Service initialized\n")

        # Option 1: Use the complete workflow (recommended)
        print("Testing complete workflow...\n")
        final_video = service.process_video(
            video_url=test_video_url,
            output_path="output/test_with_subtitles.mp4",
            title="Test Video",
            language="en",
            template_name="Hormozi 2"
        )
        print(f"\n✅ Complete workflow successful!")
        print(f"   Final video: {final_video}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
