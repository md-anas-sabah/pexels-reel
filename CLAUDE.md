# AI Reel Generation Agent - Complete Implementation Plan

## Project Overview

This document provides a comprehensive implementation plan for refactoring and expanding an existing Pexels-based reel generator into a multi-scenario AI Reel Generation Agent. The system will support three distinct workflows using Pexels, HeyGen, and Submagic APIs with intelligent orchestration via a central dispatcher.

### Three Workflow Types:

1. **Local Media Workflow**: Process and edit videos from user's computer
2. **Pexels Stock Workflow**: Generate reels from stock footage with music (no avatars)
3. **HeyGen Avatar + B-Roll Workflow** ⭐ **ENHANCED**:
   - Generate AI avatar speaking the user's script
   - Automatically fetch relevant B-roll stock footage from Pexels
   - Composite avatar over B-roll in picture-in-picture style
   - Add professional captions and effects via Submagic
   - **Result**: YouTube-style explainer videos with avatar + engaging background

---

## Architecture Overview

### Tech Stack
- **Language**: Python 3.12.7
- **Framework**: CrewAI (for multi-agent orchestration)
- **Video Processing**: FFmpeg (via `ffmpeg-python`)
- **APIs**:
  - Pexels API (stock video sourcing)
  - HeyGen API (avatar-based videos)
  - Submagic API (automated video editing)
  - Fal AI API (audio generation & text-to-speech)
- **Environment Management**: `python-dotenv`

### Project Structure
```
reel_generator/
├── main.py                          # Entry point & workflow dispatcher
├── requirements.txt                 # Python dependencies
├── .env                            # API keys (gitignored)
├── .gitignore
├── README.md
├── config.py                       # Non-secret configuration
├── services/
│   ├── __init__.py
│   ├── pexels_service.py           # Pexels API client
│   ├── heygen_service.py           # HeyGen API client
│   ├── submagic_service.py         # Submagic API client
│   └── audio_service.py            # Fal AI integration
├── workflows/
│   ├── __init__.py
│   ├── local_media_workflow.py     # Scenario 1: Local videos
│   ├── pexels_workflow.py          # Scenario 2: Pexels stock
│   └── heygen_workflow.py          # Scenario 3: Avatar videos
├── utils/
│   ├── __init__.py
│   ├── video_processor.py          # FFmpeg wrapper
│   └── file_uploader.py            # Public URL generator
└── output/
    └── [generated_reels]/
```

### Architectural Flow

```
User Input → main.py (Dispatcher) → Workflow Selection
                                         ↓
        ┌────────────────────────────────┼────────────────────────────────┐
        ↓                                ↓                                ↓
  Local Media                      Pexels Stock              HeyGen Avatar + B-Roll
  Workflow                         Workflow                  Workflow
        ↓                                ↓                                ↓
  Video Processor                  Pexels Service           HeyGen Service (Avatar)
        ↓                                ↓                          +
  File Uploader                    Video Processor          Pexels Service (B-Roll)
        ↓                                ↓                                ↓
        |                          File Uploader            Video Processor (Composite)
        |                                ↓                                ↓
        |                                |                         File Uploader
        |                                |                                ↓
        └────────────────────────────────┼────────────────────────────────┘
                                         ↓
                                 Submagic Service
                              (Captions + Effects)
                                         ↓
                                  Final Output
                                  (MP4 Reel)
```

### Critical Architecture Challenge: Inter-Service Video Transfer

**Problem**: Both HeyGen and Submagic APIs require **public URLs** for video processing. Videos generated locally or by HeyGen need temporary public hosting before submission to Submagic.

**Solution**: Implement `utils/file_uploader.py` service:
- **Development**: Use file-sharing services (e.g., file.io, tmpfiles.org)
- **Production**: AWS S3, Google Cloud Storage, or Azure Blob Storage with signed URLs

---

## Implementation Plan

### PHASE 1: Project Restructuring & Foundation

**Duration**: Days 1-2
**Goal**: Create a secure, scalable foundation with proper project structure

#### 1.1 Initialize Project Structure

**Tasks**:
```bash
# Create directory structure
mkdir -p reel_generator/{services,workflows,utils,output}
cd reel_generator

# Initialize Python environment
python3.12 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create .gitignore
cat > .gitignore << EOF
.env
venv/
__pycache__/
*.pyc
output/
*.mp4
*.mov
*.avi
.DS_Store
EOF

# Initialize git
git init
```

#### 1.2 Secure API Keys in `.env`

**File**: `.env`
```env
# API Keys
PEXELS_API_KEY=your_pexels_key_here
HEYGEN_API_KEY=your_heygen_key_here
SUBMAGIC_API_KEY=your_submagic_key_here
FAL_AI_API_KEY=your_fal_ai_key_here

# File Storage (for production)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET=

# Configuration
OUTPUT_DIR=output
MAX_VIDEO_DURATION=60
```

**File**: `config.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (loaded from environment)
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
    SUBMAGIC_API_KEY = os.getenv("SUBMAGIC_API_KEY")
    FAL_AI_API_KEY = os.getenv("FAL_AI_API_KEY")

    # HeyGen Avatar IDs
    HEYGEN_AVATARS = [
        {"id": "Angela-inblackskirt-20220820", "name": "Angela (Professional Female)"},
        {"id": "josh-incasualtshirt-20220820", "name": "Josh (Casual Male)"},
        {"id": "monica-inpinkskirt-20220820", "name": "Monica (Business)"},
        {"id": "wayne-incasualsuit-20220820", "name": "Wayne (Corporate)"},
    ]

    # Video Processing Settings
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
    MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 60))
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv"]

    # Submagic Settings
    SUBMAGIC_ENABLE_SUBTITLES = True  # Will be False for Pexels workflow
```

#### 1.3 Create `requirements.txt`

```txt
# Core dependencies
python-dotenv==1.0.0
requests==2.31.0

# Video processing
ffmpeg-python==0.2.0

# Async support
aiohttp==3.9.1
asyncio==3.4.3

# CrewAI (if using agents)
crewai==0.1.0

# Utilities
pydantic==2.5.0
tqdm==4.66.1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

#### 1.4 Refactor Existing Pexels Code

**Original**: `video_reel_converter.py` (monolithic)
**Refactor into**:
- `services/pexels_service.py` - API client
- `services/audio_service.py` - Fal AI integration
- `utils/video_processor.py` - FFmpeg operations

---

### PHASE 2: Core Service Implementation

**Duration**: Days 3-5
**Goal**: Build and test API clients for HeyGen, Submagic, and File Uploader

#### 2.1 Pexels Service (Refactored)

**File**: `services/pexels_service.py`

```python
import requests
from config import Config

class PexelsService:
    BASE_URL = "https://api.pexels.com/videos"

    def __init__(self):
        self.api_key = Config.PEXELS_API_KEY
        self.headers = {"Authorization": self.api_key}

    def search_videos(self, query: str, per_page: int = 5) -> list:
        """Search for videos on Pexels."""
        url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "portrait"  # For social media reels
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json().get("videos", [])

    def download_video(self, video_url: str, output_path: str):
        """Download video from Pexels."""
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path
```

#### 2.2 HeyGen Service

**File**: `services/heygen_service.py`

```python
import requests
import time
from config import Config

class HeyGenService:
    BASE_URL = "https://api.heygen.com/v2/video"

    def __init__(self):
        self.api_key = Config.HEYGEN_API_KEY
        self.headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def generate_video(self, script: str, avatar_id: str) -> dict:
        """
        Generate an avatar video with HeyGen.

        Args:
            script: The text the avatar will speak
            avatar_id: HeyGen avatar identifier

        Returns:
            dict with video_id for polling
        """
        url = f"{self.BASE_URL}/generate"

        payload = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": script,
                    "voice_id": "en-US-JennyNeural"  # Default voice
                },
                "background": {
                    "type": "color",
                    "value": "#FFFFFF"
                }
            }],
            "dimension": {
                "width": 1080,
                "height": 1920  # Portrait for reels
            },
            "aspect_ratio": "9:16"
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

    def get_video_status(self, video_id: str) -> dict:
        """Poll HeyGen for video generation status."""
        url = f"{self.BASE_URL}/{video_id}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def wait_for_completion(self, video_id: str, timeout: int = 300) -> str:
        """
        Wait for video generation to complete.

        Returns:
            Public URL of the generated video
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_video_status(video_id)

            if status["status"] == "completed":
                return status["video_url"]
            elif status["status"] == "failed":
                raise Exception(f"Video generation failed: {status.get('error')}")

            time.sleep(10)  # Poll every 10 seconds

        raise TimeoutError("Video generation timed out")
```

#### 2.3 Submagic Service

**File**: `services/submagic_service.py`

```python
import requests
import time
from config import Config

class SubmagicService:
    BASE_URL = "https://api.submagic.co/v1"

    def __init__(self):
        self.api_key = Config.SUBMAGIC_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def edit_video(self, video_url: str, enable_subtitles: bool = True) -> dict:
        """
        Submit a video to Submagic for editing.

        Args:
            video_url: Public URL of the video to edit
            enable_subtitles: Whether to add captions

        Returns:
            Job ID for polling
        """
        url = f"{self.BASE_URL}/videos/create"

        payload = {
            "video_url": video_url,
            "settings": {
                "add_captions": enable_subtitles,
                "caption_style": "modern",
                "add_transitions": True,
                "add_music": False,  # We handle music separately
                "aspect_ratio": "9:16"
            }
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

    def get_job_status(self, job_id: str) -> dict:
        """Check the status of a Submagic editing job."""
        url = f"{self.BASE_URL}/videos/{job_id}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def wait_for_completion(self, job_id: str, timeout: int = 600) -> str:
        """
        Wait for Submagic editing to complete.

        Returns:
            Download URL of the edited video
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status["download_url"]
            elif status["status"] == "failed":
                raise Exception(f"Submagic editing failed: {status.get('error')}")

            time.sleep(15)  # Poll every 15 seconds

        raise TimeoutError("Submagic editing timed out")

    def download_video(self, download_url: str, output_path: str):
        """Download the edited video."""
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path
```

#### 2.4 Audio Service (Fal AI)

**File**: `services/audio_service.py`

```python
import requests
from config import Config

class AudioService:
    BASE_URL = "https://api.fal.ai/v1"

    def __init__(self):
        self.api_key = Config.FAL_AI_API_KEY
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate_music(self, prompt: str, duration: int = 30) -> str:
        """Generate background music using Fal AI."""
        url = f"{self.BASE_URL}/audio/generate"

        payload = {
            "prompt": prompt,
            "duration": duration,
            "style": "upbeat"
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["audio_url"]

    def text_to_speech(self, text: str, voice: str = "en-US-neural") -> str:
        """Convert text to speech."""
        url = f"{self.BASE_URL}/tts/generate"

        payload = {
            "text": text,
            "voice_id": voice
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["audio_url"]
```

#### 2.5 File Uploader Utility

**File**: `utils/file_uploader.py`

```python
import requests
import os

class FileUploader:
    """
    Utility to upload local files and get public URLs.
    For development: Uses temporary file hosting.
    For production: Should use AWS S3 or similar.
    """

    @staticmethod
    def upload_temp(file_path: str) -> str:
        """
        Upload file to temporary hosting (development only).
        Uses file.io for 1-time download links.
        """
        url = "https://file.io"

        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            response.raise_for_status()

        return response.json()["link"]

    @staticmethod
    def upload_s3(file_path: str, bucket: str) -> str:
        """
        Upload to AWS S3 (production).
        Returns a signed URL valid for 1 hour.
        """
        import boto3
        from botocore.config import Config as BotoConfig

        s3_client = boto3.client(
            's3',
            config=BotoConfig(signature_version='s3v4')
        )

        file_name = os.path.basename(file_path)

        # Upload file
        s3_client.upload_file(file_path, bucket, file_name)

        # Generate signed URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': file_name},
            ExpiresIn=3600  # 1 hour
        )

        return url
```

#### 2.6 Video Processor Utility

**File**: `utils/video_processor.py`

```python
import ffmpeg
import os
from pathlib import Path
from config import Config

class VideoProcessor:
    """FFmpeg wrapper for video manipulation."""

    @staticmethod
    def trim_video(input_path: str, output_path: str, duration: int = 10):
        """Trim video to specified duration."""
        (
            ffmpeg
            .input(input_path, t=duration)
            .output(output_path, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path

    @staticmethod
    def concatenate_videos(video_paths: list, output_path: str):
        """Concatenate multiple videos into one."""
        # Create temporary file list
        with open("temp_filelist.txt", "w") as f:
            for path in video_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")

        # Concatenate
        (
            ffmpeg
            .input("temp_filelist.txt", format='concat', safe=0)
            .output(output_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )

        os.remove("temp_filelist.txt")
        return output_path

    @staticmethod
    def add_audio(video_path: str, audio_path: str, output_path: str):
        """Add background audio to video."""
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)

        (
            ffmpeg
            .output(video, audio, output_path, vcodec='copy', acodec='aac', shortest=None)
            .overwrite_output()
            .run(quiet=True)
        )

        return output_path

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Extract video metadata."""
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

        return {
            "duration": float(probe['format']['duration']),
            "width": int(video_info['width']),
            "height": int(video_info['height']),
            "fps": eval(video_info['r_frame_rate'])
        }

    @staticmethod
    def composite_avatar_with_broll(broll_path: str, avatar_path: str,
                                     output_path: str, position: str = "bottom-right"):
        """
        Composite avatar video over B-roll footage (picture-in-picture).

        Args:
            broll_path: Background B-roll video
            avatar_path: Avatar video to overlay
            output_path: Output composite video
            position: Avatar position - "bottom-right", "bottom-left", "top-right", "top-left", "center"

        Creates a professional YouTube-style video with:
        - B-roll as full background
        - Avatar in corner (scaled to 30% of frame)
        - Slight padding and optional border
        """
        # Get video dimensions
        broll_info = VideoProcessor.get_video_info(broll_path)
        broll_width = broll_info['width']
        broll_height = broll_info['height']

        # Calculate avatar size (30% of B-roll width)
        avatar_scale_width = int(broll_width * 0.3)

        # Define position coordinates with padding
        padding = 20
        positions = {
            "bottom-right": f"W-w-{padding}:H-h-{padding}",
            "bottom-left": f"{padding}:H-h-{padding}",
            "top-right": f"W-w-{padding}:{padding}",
            "top-left": f"{padding}:{padding}",
            "center": f"(W-w)/2:(H-h)/2"
        }

        overlay_position = positions.get(position, positions["bottom-right"])

        # Load inputs
        broll = ffmpeg.input(broll_path)
        avatar = ffmpeg.input(avatar_path)

        # Scale avatar to appropriate size
        avatar_scaled = avatar.filter('scale', avatar_scale_width, -1)

        # Optional: Add border/shadow to avatar for better visibility
        # avatar_with_border = avatar_scaled.filter('pad',
        #     avatar_scale_width + 4, 'ih+4', 2, 2, color='white')

        # Overlay avatar on B-roll
        composite = ffmpeg.overlay(
            broll,
            avatar_scaled,
            x=overlay_position.split(':')[0],
            y=overlay_position.split(':')[1],
            shortest=True  # End when shortest input ends
        )

        # Output with re-encoding
        output = ffmpeg.output(
            composite,
            output_path,
            vcodec='libx264',
            acodec='aac',
            **{'b:v': '2M', 'b:a': '192k'}  # Good quality settings
        )

        output.overwrite_output().run(quiet=True)

        return output_path

    @staticmethod
    def create_side_by_side(video1_path: str, video2_path: str, output_path: str):
        """
        Create side-by-side comparison video (alternative layout).

        Args:
            video1_path: Left video
            video2_path: Right video
            output_path: Output video
        """
        # Load inputs
        left = ffmpeg.input(video1_path)
        right = ffmpeg.input(video2_path)

        # Stack horizontally
        joined = ffmpeg.filter([left, right], 'hstack')

        # Output
        output = ffmpeg.output(
            joined,
            output_path,
            vcodec='libx264',
            acodec='aac'
        )

        output.overwrite_output().run(quiet=True)

        return output_path
```

---

### PHASE 3: Workflow Implementation

**Duration**: Days 6-9
**Goal**: Build end-to-end logic for all three user scenarios

#### 3.1 Workflow 1: Local Media

**File**: `workflows/local_media_workflow.py`

```python
import os
from pathlib import Path
from services.submagic_service import SubmagicService
from utils.video_processor import VideoProcessor
from utils.file_uploader import FileUploader

class LocalMediaWorkflow:
    """
    Workflow for processing local video files.

    Steps:
    1. Scan local directory for video files
    2. Trim and concatenate videos
    3. Upload stitched video to get public URL
    4. Submit to Submagic for editing
    5. Download final output
    """

    def __init__(self):
        self.submagic = SubmagicService()
        self.processor = VideoProcessor()
        self.uploader = FileUploader()

    def scan_directory(self, directory: str) -> list:
        """Scan directory for video files."""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        video_files = []

        for file in Path(directory).rglob("*"):
            if file.suffix.lower() in video_extensions:
                video_files.append(str(file))

        return video_files

    def execute(self, input_directory: str, output_path: str):
        """Execute the local media workflow."""
        print("🎬 Starting Local Media Workflow...")

        # Step 1: Scan for videos
        print("📁 Scanning directory...")
        video_files = self.scan_directory(input_directory)

        if not video_files:
            raise ValueError("No video files found in directory")

        print(f"✓ Found {len(video_files)} video(s)")

        # Step 2: Trim videos to 10 seconds each
        print("✂️  Trimming videos...")
        trimmed_videos = []

        for i, video in enumerate(video_files):
            trimmed_path = f"temp_trimmed_{i}.mp4"
            self.processor.trim_video(video, trimmed_path, duration=10)
            trimmed_videos.append(trimmed_path)

        # Step 3: Concatenate videos
        print("🔗 Concatenating videos...")
        stitched_path = "temp_stitched.mp4"
        self.processor.concatenate_videos(trimmed_videos, stitched_path)

        # Step 4: Upload to get public URL
        print("☁️  Uploading video...")
        public_url = self.uploader.upload_temp(stitched_path)
        print(f"✓ Video uploaded: {public_url}")

        # Step 5: Submit to Submagic
        print("✨ Submitting to Submagic for editing...")
        job = self.submagic.edit_video(public_url, enable_subtitles=True)
        job_id = job["id"]

        print("⏳ Waiting for Submagic to complete...")
        download_url = self.submagic.wait_for_completion(job_id)

        # Step 6: Download final video
        print("📥 Downloading final video...")
        self.submagic.download_video(download_url, output_path)

        # Cleanup
        for temp_file in trimmed_videos + [stitched_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"✅ Workflow complete! Output: {output_path}")
        return output_path
```

#### 3.2 Workflow 2: Pexels Stock

**File**: `workflows/pexels_workflow.py`

```python
import os
from services.pexels_service import PexelsService
from services.audio_service import AudioService
from services.submagic_service import SubmagicService
from utils.video_processor import VideoProcessor
from utils.file_uploader import FileUploader

class PexelsWorkflow:
    """
    Workflow for creating reels from Pexels stock footage.

    Steps:
    1. Search Pexels for relevant videos
    2. Download videos
    3. Trim and concatenate
    4. Add background music
    5. Upload to get public URL
    6. Submit to Submagic (NO subtitles for this workflow)
    7. Download final output
    """

    def __init__(self):
        self.pexels = PexelsService()
        self.audio = AudioService()
        self.submagic = SubmagicService()
        self.processor = VideoProcessor()
        self.uploader = FileUploader()

    def execute(self, topic: str, output_path: str):
        """Execute the Pexels workflow."""
        print(f"🎬 Starting Pexels Workflow for topic: '{topic}'")

        # Step 1: Search Pexels
        print("🔍 Searching Pexels for videos...")
        videos = self.pexels.search_videos(topic, per_page=5)

        if not videos:
            raise ValueError(f"No videos found for topic: {topic}")

        print(f"✓ Found {len(videos)} videos")

        # Step 2: Download videos
        print("📥 Downloading videos...")
        downloaded_videos = []

        for i, video in enumerate(videos):
            # Get the HD video file
            video_file = next((f for f in video['video_files'] if f['quality'] == 'hd'), video['video_files'][0])
            video_url = video_file['link']

            download_path = f"temp_pexels_{i}.mp4"
            self.pexels.download_video(video_url, download_path)
            downloaded_videos.append(download_path)

        # Step 3: Trim videos
        print("✂️  Trimming videos...")
        trimmed_videos = []

        for i, video in enumerate(downloaded_videos):
            trimmed_path = f"temp_trimmed_{i}.mp4"
            self.processor.trim_video(video, trimmed_path, duration=6)
            trimmed_videos.append(trimmed_path)

        # Step 4: Concatenate
        print("🔗 Concatenating videos...")
        stitched_path = "temp_pexels_stitched.mp4"
        self.processor.concatenate_videos(trimmed_videos, stitched_path)

        # Step 5: Add background music
        print("🎵 Generating background music...")
        music_url = self.audio.generate_music(f"upbeat music for {topic}", duration=30)

        # Download music
        import requests
        music_path = "temp_music.mp3"
        with open(music_path, "wb") as f:
            f.write(requests.get(music_url).content)

        print("🎵 Adding music to video...")
        video_with_audio = "temp_pexels_with_audio.mp4"
        self.processor.add_audio(stitched_path, music_path, video_with_audio)

        # Step 6: Upload to get public URL
        print("☁️  Uploading video...")
        public_url = self.uploader.upload_temp(video_with_audio)

        # Step 7: Submit to Submagic (NO SUBTITLES)
        print("✨ Submitting to Submagic for editing...")
        job = self.submagic.edit_video(public_url, enable_subtitles=False)
        job_id = job["id"]

        print("⏳ Waiting for Submagic to complete...")
        download_url = self.submagic.wait_for_completion(job_id)

        # Step 8: Download final video
        print("📥 Downloading final video...")
        self.submagic.download_video(download_url, output_path)

        # Cleanup
        temp_files = downloaded_videos + trimmed_videos + [stitched_path, music_path, video_with_audio]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"✅ Workflow complete! Output: {output_path}")
        return output_path
```

#### 3.3 Workflow 3: HeyGen Avatar + Stock B-Roll

**File**: `workflows/heygen_workflow.py`

```python
import os
import requests
from services.heygen_service import HeyGenService
from services.pexels_service import PexelsService
from services.submagic_service import SubmagicService
from utils.video_processor import VideoProcessor
from utils.file_uploader import FileUploader
from config import Config

class HeyGenWorkflow:
    """
    Workflow for creating avatar-based reels with HeyGen + Stock B-Roll.

    Steps:
    1. User selects an avatar
    2. User provides script and B-roll topic
    3. Generate avatar video with HeyGen
    4. Download stock B-roll footage from Pexels
    5. Composite avatar over B-roll (picture-in-picture style)
    6. Upload composite video to get public URL
    7. Submit to Submagic for final editing (captions, effects)
    8. Download final output
    """

    def __init__(self):
        self.heygen = HeyGenService()
        self.pexels = PexelsService()
        self.submagic = SubmagicService()
        self.processor = VideoProcessor()
        self.uploader = FileUploader()

    def display_avatars(self):
        """Display available avatars for user selection."""
        print("\n=== Available Avatars ===")
        for i, avatar in enumerate(Config.HEYGEN_AVATARS, 1):
            print(f"{i}. {avatar['name']} (ID: {avatar['id']})")
        print()

    def select_avatar(self) -> str:
        """Prompt user to select an avatar."""
        self.display_avatars()

        while True:
            try:
                choice = int(input("Select avatar number: ")) - 1
                if 0 <= choice < len(Config.HEYGEN_AVATARS):
                    return Config.HEYGEN_AVATARS[choice]["id"]
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")

    def extract_keywords_from_script(self, script: str) -> str:
        """
        Extract keywords from script for B-roll search.
        Simple implementation - can be enhanced with NLP.
        """
        # Remove common words and take first few meaningful words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = script.lower().split()
        keywords = [w for w in words if w not in common_words and len(w) > 3]

        # Return first 2-3 keywords as search query
        return ' '.join(keywords[:3]) if keywords else "technology business"

    def execute(self, script: str, avatar_id: str, broll_topic: str, output_path: str):
        """Execute the HeyGen + B-roll workflow."""
        print(f"🎬 Starting HeyGen Avatar + B-Roll Workflow")
        print(f"   Avatar: {avatar_id}")
        print(f"   B-Roll Topic: {broll_topic}")

        # Step 1: Generate avatar video
        print("\n🤖 Generating avatar video with HeyGen...")
        result = self.heygen.generate_video(script, avatar_id)
        video_id = result["video_id"]

        print(f"✓ Video generation started (ID: {video_id})")

        # Step 2: While avatar is generating, fetch B-roll from Pexels
        print("\n🔍 Searching Pexels for B-roll footage...")
        broll_videos = self.pexels.search_videos(broll_topic, per_page=3)

        if not broll_videos:
            print(f"⚠️  No B-roll found for '{broll_topic}', using default search")
            broll_videos = self.pexels.search_videos("abstract background", per_page=3)

        print(f"✓ Found {len(broll_videos)} B-roll videos")

        # Step 3: Download B-roll videos
        print("📥 Downloading B-roll videos...")
        downloaded_broll = []

        for i, video in enumerate(broll_videos):
            video_file = next((f for f in video['video_files'] if f['quality'] == 'hd'), video['video_files'][0])
            video_url = video_file['link']

            download_path = f"temp_broll_{i}.mp4"
            self.pexels.download_video(video_url, download_path)
            downloaded_broll.append(download_path)

        # Step 4: Trim and concatenate B-roll
        print("✂️  Processing B-roll footage...")
        trimmed_broll = []

        for i, broll in enumerate(downloaded_broll):
            trimmed_path = f"temp_broll_trimmed_{i}.mp4"
            self.processor.trim_video(broll, trimmed_path, duration=10)
            trimmed_broll.append(trimmed_path)

        broll_stitched = "temp_broll_stitched.mp4"
        self.processor.concatenate_videos(trimmed_broll, broll_stitched)

        # Step 5: Wait for HeyGen avatar to complete
        print("\n⏳ Waiting for HeyGen to generate avatar (this may take a few minutes)...")
        heygen_url = self.heygen.wait_for_completion(video_id)

        print(f"✓ Avatar video ready: {heygen_url}")

        # Step 6: Download avatar video
        print("📥 Downloading avatar video...")
        avatar_path = "temp_avatar.mp4"
        with open(avatar_path, "wb") as f:
            f.write(requests.get(heygen_url).content)

        # Step 7: Composite avatar over B-roll
        print("🎨 Compositing avatar over B-roll (picture-in-picture)...")
        composite_path = "temp_composite.mp4"
        self.processor.composite_avatar_with_broll(
            broll_path=broll_stitched,
            avatar_path=avatar_path,
            output_path=composite_path,
            position="bottom-right"  # Avatar in corner, B-roll as background
        )

        # Step 8: Upload composite to get public URL
        print("☁️  Uploading composite video...")
        public_url = self.uploader.upload_temp(composite_path)
        print(f"✓ Video uploaded: {public_url}")

        # Step 9: Submit to Submagic for final polish
        print("\n✨ Submitting to Submagic for final editing (captions, effects)...")
        job = self.submagic.edit_video(public_url, enable_subtitles=True)
        job_id = job["id"]

        print("⏳ Waiting for Submagic to complete...")
        download_url = self.submagic.wait_for_completion(job_id)

        # Step 10: Download final video
        print("📥 Downloading final video...")
        self.submagic.download_video(download_url, output_path)

        # Cleanup
        print("🧹 Cleaning up temporary files...")
        temp_files = (downloaded_broll + trimmed_broll +
                     [broll_stitched, avatar_path, composite_path])

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"\n✅ Workflow complete! Output: {output_path}")
        print(f"   🎥 Professional avatar video with B-roll background")
        print(f"   📝 Subtitles and effects added by Submagic")

        return output_path
```

---

### PHASE 4: Integration & Main Dispatcher

**Duration**: Days 10-11
**Goal**: Build the central dispatcher and integrate all workflows

#### 4.1 Main Dispatcher

**File**: `main.py`

```python
#!/usr/bin/env python3
"""
AI Reel Generation Agent - Main Entry Point

This script serves as the central dispatcher for three workflow scenarios:
1. Local Media: Process videos from a local directory
2. Pexels Stock: Generate reels from Pexels stock footage
3. HeyGen Avatar + B-Roll: Create professional avatar reels with stock footage background
"""

import os
from datetime import datetime
from pathlib import Path
from workflows.local_media_workflow import LocalMediaWorkflow
from workflows.pexels_workflow import PexelsWorkflow
from workflows.heygen_workflow import HeyGenWorkflow
from config import Config

def display_menu():
    """Display the main menu."""
    print("\n" + "="*50)
    print("🎬 AI REEL GENERATION AGENT")
    print("="*50)
    print("\nSelect a workflow:\n")
    print("1. Local Media Workflow")
    print("   → Process videos from your computer")
    print()
    print("2. Pexels Stock Workflow")
    print("   → Generate reels from stock footage")
    print()
    print("3. HeyGen Avatar + B-Roll Workflow")
    print("   → Create professional avatar reels with stock footage background")
    print()
    print("4. Exit")
    print()

def create_output_path(workflow_name: str) -> str:
    """Create timestamped output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    return str(output_dir / f"{workflow_name}_{timestamp}.mp4")

def run_local_media_workflow():
    """Execute Local Media Workflow."""
    print("\n--- Local Media Workflow ---")

    directory = input("Enter path to video directory: ").strip()

    if not os.path.isdir(directory):
        print("❌ Invalid directory path.")
        return

    output_path = create_output_path("local_media")

    try:
        workflow = LocalMediaWorkflow()
        workflow.execute(directory, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")

def run_pexels_workflow():
    """Execute Pexels Stock Workflow."""
    print("\n--- Pexels Stock Workflow ---")

    topic = input("Enter topic for stock footage (e.g., 'ocean waves', 'city life'): ").strip()

    if not topic:
        print("❌ Topic cannot be empty.")
        return

    output_path = create_output_path("pexels")

    try:
        workflow = PexelsWorkflow()
        workflow.execute(topic, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")

def run_heygen_workflow():
    """Execute HeyGen Avatar + B-Roll Workflow."""
    print("\n--- HeyGen Avatar + B-Roll Workflow ---")

    workflow = HeyGenWorkflow()

    # Avatar selection
    avatar_id = workflow.select_avatar()

    # Script input
    print("\nEnter your script (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break
            lines.append(line)
        else:
            lines.append(line)

    script = "\n".join(lines).strip()

    if not script:
        print("❌ Script cannot be empty.")
        return

    # B-roll topic input
    print("\nEnter B-roll topic for background footage:")
    print("(e.g., 'technology', 'business meeting', 'nature', 'city life')")
    broll_topic = input("B-roll topic: ").strip()

    if not broll_topic:
        # Extract keywords from script if user doesn't provide topic
        print("⚠️  No B-roll topic provided, extracting from script...")
        broll_topic = workflow.extract_keywords_from_script(script)
        print(f"   Using: '{broll_topic}'")

    output_path = create_output_path("heygen_broll")

    try:
        workflow.execute(script, avatar_id, broll_topic, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main entry point."""
    while True:
        display_menu()

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            run_local_media_workflow()
        elif choice == "2":
            run_pexels_workflow()
        elif choice == "3":
            run_heygen_workflow()
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Verify API keys are set
    if not Config.PEXELS_API_KEY:
        print("❌ Error: PEXELS_API_KEY not found in .env file")
        exit(1)

    if not Config.HEYGEN_API_KEY:
        print("⚠️  Warning: HEYGEN_API_KEY not set (HeyGen workflow will fail)")

    if not Config.SUBMAGIC_API_KEY:
        print("⚠️  Warning: SUBMAGIC_API_KEY not set (Submagic editing will fail)")

    main()
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_services.py`

```python
import unittest
from unittest.mock import Mock, patch
from services.pexels_service import PexelsService
from services.heygen_service import HeyGenService

class TestPexelsService(unittest.TestCase):
    @patch('services.pexels_service.requests.get')
    def test_search_videos(self, mock_get):
        mock_get.return_value.json.return_value = {
            "videos": [{"id": 1}, {"id": 2}]
        }

        service = PexelsService()
        videos = service.search_videos("ocean")

        self.assertEqual(len(videos), 2)

class TestHeyGenService(unittest.TestCase):
    @patch('services.heygen_service.requests.post')
    def test_generate_video(self, mock_post):
        mock_post.return_value.json.return_value = {
            "video_id": "test123"
        }

        service = HeyGenService()
        result = service.generate_video("Hello world", "avatar_001")

        self.assertEqual(result["video_id"], "test123")

if __name__ == "__main__":
    unittest.main()
```

### Integration Tests

**File**: `tests/test_workflows.py`

```python
import unittest
import os
from workflows.local_media_workflow import LocalMediaWorkflow

class TestLocalMediaWorkflow(unittest.TestCase):
    def setUp(self):
        # Create test video directory
        os.makedirs("test_videos", exist_ok=True)

    def tearDown(self):
        # Cleanup
        import shutil
        if os.path.exists("test_videos"):
            shutil.rmtree("test_videos")

    def test_scan_directory(self):
        # Create dummy video file
        test_file = "test_videos/test.mp4"
        with open(test_file, "w") as f:
            f.write("dummy")

        workflow = LocalMediaWorkflow()
        videos = workflow.scan_directory("test_videos")

        self.assertEqual(len(videos), 1)

if __name__ == "__main__":
    unittest.main()
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All API keys configured in `.env`
- [ ] FFmpeg installed on system (`brew install ffmpeg` or `apt install ffmpeg`)
- [ ] Python 3.12.7 installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test videos available for local workflow testing

### Testing
- [ ] Pexels API key validated
- [ ] HeyGen API key validated
- [ ] Submagic API key validated
- [ ] Fal AI API key validated
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Test each workflow with real API calls

### Production Considerations
- [ ] Replace `file_uploader.py` temp hosting with AWS S3
- [ ] Implement error logging (use Python `logging` module)
- [ ] Add rate limiting for API calls
- [ ] Implement retry logic with exponential backoff
- [ ] Set up monitoring and alerts
- [ ] Create user documentation

---

## File Uploader Production Setup

### AWS S3 Configuration

1. **Install boto3**:
```bash
pip install boto3
```

2. **Add to `.env`**:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
```

3. **Update `config.py`**:
```python
# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
```

4. **Update `file_uploader.py`** to use S3 by default in production

---

## Troubleshooting Guide

### Common Issues

**Issue**: FFmpeg not found
```
Solution: Install FFmpeg
  macOS: brew install ffmpeg
  Ubuntu: sudo apt install ffmpeg
  Windows: Download from ffmpeg.org
```

**Issue**: File upload fails (file.io)
```
Solution:
  - Check internet connection
  - Verify file size < 100MB
  - Switch to S3 for production
```

**Issue**: HeyGen video generation timeout
```
Solution:
  - Increase timeout in wait_for_completion()
  - Check HeyGen API status
  - Verify avatar_id is valid
```

**Issue**: Submagic job fails
```
Solution:
  - Ensure video URL is publicly accessible
  - Check video format (MP4 recommended)
  - Verify API key permissions
```

**Issue**: API rate limiting
```
Solution:
  - Implement exponential backoff
  - Add delays between requests
  - Check API plan limits
```

---

## Future Enhancements

1. **Web Interface**
   - Flask/FastAPI dashboard
   - Drag-and-drop file upload
   - Real-time progress tracking

2. **Batch Processing**
   - Process multiple topics simultaneously
   - Queue system for workflows

3. **Advanced Editing**
   - Custom Submagic presets
   - User-defined caption styles
   - Transition customization

4. **Analytics**
   - Track processing times
   - Monitor API usage
   - Cost analysis

5. **AI Enhancements**
   - Auto-generate scripts with GPT
   - Content moderation
   - Automatic topic suggestions

6. **Multi-Platform Export**
   - Instagram Reels (1080x1920)
   - TikTok (1080x1920)
   - YouTube Shorts (1080x1920)
   - Custom aspect ratios

---

## Project Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | Days 1-2 | Project structure, config, refactored code |
| Phase 2 | Days 3-5 | All service implementations |
| Phase 3 | Days 6-9 | Three workflow implementations |
| Phase 4 | Days 10-11 | Main dispatcher, integration |
| Testing | Day 12 | Unit & integration tests |
| Documentation | Day 13 | README, API docs |
| Deployment | Day 14 | Production setup, final testing |

**Total**: 14 days (2 weeks)

---

## Conclusion

This implementation plan provides a complete roadmap for building a production-ready AI Reel Generation Agent with three distinct workflows. The modular architecture ensures scalability, maintainability, and easy extension for future features.

Key architectural decisions:
- **Dispatcher pattern** for clean workflow separation
- **Service layer** for API abstraction
- **Utility layer** for shared functionality
- **File uploader** solves the critical inter-service URL requirement

The system is designed to be developer-friendly with clear separation of concerns and follows Python best practices throughout.
