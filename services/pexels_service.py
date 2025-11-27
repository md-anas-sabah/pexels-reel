"""
Pexels API Service
Handles video search and download from Pexels stock library
"""

import os
import requests
import logging
from typing import Dict, List, Optional
from pathlib import Path
from config import Config


logger = logging.getLogger(__name__)


class PexelsService:
    """Service for interacting with Pexels API to search and download videos"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pexels service

        Args:
            api_key: Pexels API key (defaults to Config.PEXELS_API_KEY)
        """
        self.api_key = api_key or Config.PEXELS_API_KEY
        self.base_url = Config.PEXELS_BASE_URL

        if not self.api_key:
            raise ValueError("Pexels API key is required")

        self.headers = {
            "Authorization": self.api_key
        }

    def search_videos(
        self,
        query: str,
        per_page: int = 5,
        min_width: int = 1280,
        orientation: str = "portrait"
    ) -> List[Dict]:
        """
        Search for videos on Pexels

        Args:
            query: Search query keyword(s)
            per_page: Number of videos to fetch (default: 5)
            min_width: Minimum video width (default: 1280 for HD)
            orientation: Video orientation - "portrait", "landscape", or "square" (default: "portrait")

        Returns:
            List of video data dictionaries
        """
        params = {
            "query": query,
            "per_page": per_page,
            "min_width": min_width,
            "orientation": orientation
        }

        print(f"\n🔍 Searching Pexels for: '{query}'")
        print(f"   Orientation: {orientation}, Min Width: {min_width}px")

        try:
            response = requests.get(
                f"{self.base_url}/search",
                headers=self.headers,
                params=params,
                timeout=30
            )

            # Log rate limit info
            limit = response.headers.get('X-Ratelimit-Limit', 'unknown')
            remaining = response.headers.get('X-Ratelimit-Remaining', 'unknown')
            reset = response.headers.get('X-Ratelimit-Reset', 'unknown')

            print(f"   📊 Rate Limit: {remaining}/{limit} remaining")

            if response.status_code == 429:
                raise Exception(f"Pexels API rate limit exceeded. Resets at: {reset}")

            response.raise_for_status()
            data = response.json()

            videos = data.get("videos", [])
            total_results = data.get("total_results", 0)

            print(f"   ✅ Found {len(videos)} videos (Total available: {total_results})")

            # Extract relevant video information
            video_list = []
            for video in videos:
                video_files = video.get("video_files", [])

                # Find best quality video file
                best_file = self._select_best_quality(video_files)

                if best_file:
                    video_info = {
                        "id": video.get("id"),
                        "width": video.get("width"),
                        "height": video.get("height"),
                        "duration": video.get("duration"),
                        "url": video.get("url"),
                        "download_url": best_file.get("link"),
                        "quality": best_file.get("quality"),
                        "photographer": video.get("user", {}).get("name", "Unknown")
                    }
                    video_list.append(video_info)

            return video_list

        except requests.exceptions.RequestException as e:
            logger.error(f"Pexels API request failed: {str(e)}")
            raise

    def _select_best_quality(self, video_files: List[Dict]) -> Optional[Dict]:
        """
        Select the best quality video file from available options

        Args:
            video_files: List of video file dictionaries from Pexels API

        Returns:
            Best quality video file dict or None
        """
        if not video_files:
            return None

        # Preferred qualities in order
        quality_preference = ["hd", "sd"]

        # Try to find video with preferred quality
        for quality in quality_preference:
            for video_file in video_files:
                if video_file.get("quality") == quality:
                    # Also check if it has reasonable dimensions
                    width = video_file.get("width", 0)
                    height = video_file.get("height", 0)

                    if width >= 720 and height >= 1280:  # Good for 9:16
                        return video_file

        # If no preferred quality found, return first available
        return video_files[0] if video_files else None

    def download_video(self, url: str, output_path: str) -> str:
        """
        Download video from Pexels URL

        Args:
            url: Download URL for the video
            output_path: Local path to save video

        Returns:
            Path to downloaded video
        """
        print(f"\n📥 Downloading video...")
        print(f"   URL: {url}")
        print(f"   Output: {output_path}")

        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"   Progress: {progress:.1f}%", end='\r')

            print(f"\n   ✅ Download complete! Size: {downloaded / (1024*1024):.2f} MB")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Video download failed: {str(e)}")
            raise

    def search_and_download(
        self,
        query: str,
        output_dir: str,
        count: int = 3,
        orientation: str = "portrait"
    ) -> List[str]:
        """
        Convenience method: Search for videos and download them

        Args:
            query: Search query
            output_dir: Directory to save videos
            count: Number of videos to download (default: 3)
            orientation: Video orientation (default: "portrait")

        Returns:
            List of paths to downloaded videos
        """
        print(f"\n🎬 Searching and downloading {count} videos for: '{query}'")

        # Search for videos
        videos = self.search_videos(query, per_page=count, orientation=orientation)

        if not videos:
            raise Exception(f"No videos found for query: {query}")

        # Download videos
        downloaded_paths = []
        for i, video in enumerate(videos[:count], 1):
            output_path = os.path.join(output_dir, f"pexels_video_{video['id']}.mp4")

            try:
                self.download_video(video['download_url'], output_path)
                downloaded_paths.append(output_path)
            except Exception as e:
                logger.warning(f"Failed to download video {i}: {str(e)}")
                continue

        print(f"\n✅ Successfully downloaded {len(downloaded_paths)}/{count} videos")
        return downloaded_paths

    def search_multiple_keywords(
        self,
        keywords: List[str],
        per_keyword: int = 2,
        orientation: str = "portrait"
    ) -> List[Dict]:
        """
        Search for videos using multiple keywords

        Args:
            keywords: List of search keywords
            per_keyword: Videos per keyword (default: 2)
            orientation: Video orientation (default: "portrait")

        Returns:
            Combined list of video data
        """
        print(f"\n🔍 Searching for videos with {len(keywords)} keywords...")
        all_videos = []

        for keyword in keywords:
            try:
                videos = self.search_videos(
                    query=keyword,
                    per_page=per_keyword,
                    orientation=orientation
                )
                all_videos.extend(videos)
                print(f"   ✓ '{keyword}': Found {len(videos)} videos")
            except Exception as e:
                logger.warning(f"Failed to search for '{keyword}': {str(e)}")
                continue

        # Remove duplicates by video ID
        unique_videos = []
        seen_ids = set()

        for video in all_videos:
            if video['id'] not in seen_ids:
                unique_videos.append(video)
                seen_ids.add(video['id'])

        print(f"   ✅ Total unique videos: {len(unique_videos)}")
        return unique_videos


# Example usage
if __name__ == "__main__":
    # Test Pexels service
    service = PexelsService()

    # Search for videos
    videos = service.search_videos("sunrise nature", per_page=3)

    for video in videos:
        print(f"\nVideo ID: {video['id']}")
        print(f"  Photographer: {video['photographer']}")
        print(f"  Duration: {video['duration']}s")
        print(f"  Resolution: {video['width']}x{video['height']}")

    # Download first video
    if videos:
        service.download_video(videos[0]['download_url'], "output/test_pexels.mp4")
