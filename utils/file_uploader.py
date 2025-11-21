"""
File Uploader Utility - Supabase Storage Integration

This module provides file upload functionality for getting public URLs
required by HeyGen and Submagic APIs.

Production: Uses Supabase Storage for reliable, scalable file hosting
Development: Falls back to temporary file hosting services

Author: AI Reel Generation Agent
"""

import os
import requests
import logging
from pathlib import Path
from typing import Optional
from config import Config

# Configure logging
logger = logging.getLogger(__name__)


class FileUploader:
    """
    Utility to upload local files and get public URLs.

    Uses Supabase Storage for production-grade file hosting with:
    - Automatic signed URL generation
    - Configurable expiry times
    - Large file support (up to 100MB)
    - Secure access control
    """

    def __init__(self):
        """Initialize the file uploader with Supabase credentials"""
        self.supabase_url = Config.SUPABASE_URL
        self.supabase_key = Config.SUPABASE_KEY
        self.bucket_name = Config.SUPABASE_BUCKET
        self.url_expiry = Config.SUPABASE_URL_EXPIRY
        self.max_file_size = Config.SUPABASE_MAX_FILE_SIZE

        # Flag to determine if Supabase is configured
        self.use_supabase = bool(self.supabase_url and self.supabase_key)

        if self.use_supabase:
            self._init_supabase_client()
        else:
            logger.warning("Supabase credentials not configured. Falling back to temporary file hosting.")

    def _init_supabase_client(self):
        """Initialize Supabase client"""
        try:
            from supabase import create_client, Client

            # Simple initialization - let the library handle defaults
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"✓ Supabase client initialized for bucket: {self.bucket_name}")

        except ImportError:
            logger.error("Supabase library not installed. Run: pip install supabase")
            self.use_supabase = False

        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            logger.warning("Will fall back to temporary file hosting")
            self.use_supabase = False

    def upload(self, file_path: str) -> str:
        """
        Upload file and return a public URL.

        Automatically chooses between Supabase (production) and
        temporary hosting (development) based on configuration.

        Args:
            file_path: Absolute path to the file to upload

        Returns:
            Public URL of the uploaded file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large
            Exception: If upload fails
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.2f}MB "
                f"(max: {self.max_file_size / 1024 / 1024:.2f}MB)"
            )

        # Use appropriate upload method
        if self.use_supabase:
            return self._upload_to_supabase(file_path)
        else:
            return self._upload_temp(file_path)

    def _upload_to_supabase(self, file_path: str) -> str:
        """
        Upload file to Supabase Storage and get a signed URL.

        Args:
            file_path: Path to file to upload

        Returns:
            Signed public URL valid for configured duration
        """
        try:
            file_name = os.path.basename(file_path)

            # Generate unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_name = f"{timestamp}_{file_name}"

            logger.info(f"📤 Uploading {file_name} to Supabase ({os.path.getsize(file_path) / 1024 / 1024:.2f}MB)...")

            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Upload to Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).upload(
                unique_name,
                file_content,
                {
                    "content-type": self._get_content_type(file_path),
                    "x-upsert": "true"  # Overwrite if exists
                }
            )

            logger.info(f"✓ File uploaded: {unique_name}")

            # Generate signed URL
            signed_url = self.supabase.storage.from_(self.bucket_name).create_signed_url(
                unique_name,
                self.url_expiry
            )

            public_url = signed_url['signedURL']

            logger.info(f"✓ Public URL generated (expires in {self.url_expiry / 3600:.1f} hours)")

            return public_url

        except Exception as e:
            logger.error(f"❌ Supabase upload failed: {e}")
            logger.warning("⚠️  Falling back to temporary file hosting...")
            return self._upload_temp(file_path)

    def _upload_temp(self, file_path: str) -> str:
        """
        Upload file to temporary hosting (development fallback).

        Uses file.io for 1-time download links.
        WARNING: This is NOT suitable for production!

        Args:
            file_path: Path to file to upload

        Returns:
            Temporary public URL (expires after first download)
        """
        logger.warning("⚠️  Using temporary file hosting (NOT for production!)")

        try:
            url = "https://file.io"

            file_name = os.path.basename(file_path)
            logger.info(f"📤 Uploading {file_name} to file.io...")

            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files)
                response.raise_for_status()

            result = response.json()

            if result.get("success"):
                public_url = result["link"]
                logger.info(f"✓ Temporary URL generated: {public_url}")
                logger.warning("⚠️  URL expires after first download!")
                return public_url
            else:
                raise Exception(f"Upload failed: {result.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Temporary upload failed: {e}")
            raise Exception(f"All upload methods failed. Error: {e}")

    def _get_content_type(self, file_path: str) -> str:
        """
        Determine MIME type based on file extension.

        Args:
            file_path: Path to file

        Returns:
            MIME type string
        """
        extension = Path(file_path).suffix.lower()

        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }

        return mime_types.get(extension, 'application/octet-stream')

    def delete_file(self, file_url: str) -> bool:
        """
        Delete a file from Supabase Storage (cleanup).

        Args:
            file_url: The signed URL or filename to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.use_supabase:
            logger.warning("Cannot delete from temporary hosting")
            return False

        try:
            # Extract filename from URL if needed
            if file_url.startswith('http'):
                file_name = file_url.split('/')[-1].split('?')[0]
            else:
                file_name = file_url

            self.supabase.storage.from_(self.bucket_name).remove([file_name])
            logger.info(f"✓ Deleted file: {file_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete file: {e}")
            return False

    def list_files(self, prefix: Optional[str] = None) -> list:
        """
        List files in Supabase Storage bucket.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file metadata dictionaries
        """
        if not self.use_supabase:
            logger.warning("File listing only available with Supabase")
            return []

        try:
            files = self.supabase.storage.from_(self.bucket_name).list(prefix)
            return files
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []


# ==================== HELPER FUNCTIONS ====================

def create_supabase_bucket_if_not_exists():
    """
    Create the Supabase storage bucket if it doesn't exist.

    This is a one-time setup function. Run this before first use.
    """
    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        print("❌ Supabase credentials not configured")
        return False

    try:
        from supabase import create_client

        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

        # Try to create bucket
        supabase.storage.create_bucket(
            Config.SUPABASE_BUCKET,
            {
                "public": False,  # Private bucket with signed URLs
                "file_size_limit": Config.SUPABASE_MAX_FILE_SIZE,
                "allowed_mime_types": [
                    "video/mp4",
                    "video/quicktime",
                    "video/x-msvideo",
                    "video/x-matroska",
                    "audio/mpeg",
                    "audio/wav"
                ]
            }
        )

        print(f"✓ Bucket '{Config.SUPABASE_BUCKET}' created successfully")
        return True

    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓ Bucket '{Config.SUPABASE_BUCKET}' already exists")
            return True
        else:
            print(f"❌ Failed to create bucket: {e}")
            return False


def test_upload(test_file_path: Optional[str] = None):
    """
    Test the file uploader functionality.

    Args:
        test_file_path: Optional path to test file. Creates dummy file if not provided.
    """
    import tempfile

    # Create test file if not provided
    if test_file_path is None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file for Supabase upload\n")
            test_file_path = f.name

    print("\n" + "="*60)
    print("TESTING FILE UPLOADER")
    print("="*60)

    try:
        uploader = FileUploader()

        print(f"\nTest file: {test_file_path}")
        print(f"File size: {os.path.getsize(test_file_path)} bytes")
        print(f"Using Supabase: {uploader.use_supabase}")

        print("\nUploading...")
        public_url = uploader.upload(test_file_path)

        print(f"\n✅ Upload successful!")
        print(f"Public URL: {public_url}")

        # Clean up test file
        if test_file_path.endswith('.txt'):
            os.remove(test_file_path)

        return public_url

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return None


if __name__ == "__main__":
    # Run test when module is executed directly
    print("Running file uploader test...")
    test_upload()
