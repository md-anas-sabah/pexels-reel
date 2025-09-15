"""
Configuration settings for Video Reel Converter
"""

import os
from pathlib import Path

# API Configuration
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_BASE_URL = "https://api.pexels.com/videos"

# Video Processing Settings
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
TARGET_ASPECT_RATIO = 9.0 / 16.0

# Quality Settings
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
CRF_VALUE = 23  # Constant Rate Factor (lower = better quality, higher file size)
PRESET = "medium"  # Encoding speed preset

# Object Detection Settings
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

BODY_SCALE_FACTOR = 1.1
BODY_MIN_NEIGHBORS = 3
BODY_MIN_SIZE = (50, 50)

# ROI (Region of Interest) Settings
ROI_PADDING_FACTOR = 0.2  # 20% padding around detected objects
SAMPLE_FRAME_INTERVAL = 20  # Analyze every 20th frame for performance

# Output Settings
OUTPUT_DIR = "output_reels"
TEMP_DIR_PREFIX = "video_reel_temp_"

# File Settings
MAX_VIDEO_DURATION = 180  # Maximum video duration in seconds (3 minutes)
MIN_VIDEO_DURATION = 5   # Minimum video duration in seconds
PREFERRED_QUALITY = ["hd", "sd"]  # Preferred video quality order

# CrewAI Settings
CREW_VERBOSE = True
CREW_PROCESS = "sequential"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Performance Settings
MAX_CONCURRENT_DOWNLOADS = 3
CHUNK_SIZE = 8192  # bytes for video download

# Validation Settings
ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
MAX_FILE_SIZE_MB = 500  # Maximum input file size in MB

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not PEXELS_API_KEY or PEXELS_API_KEY == "YOUR_API_KEY":
        errors.append("PEXELS_API_KEY must be set to a valid API key")
    
    if TARGET_WIDTH <= 0 or TARGET_HEIGHT <= 0:
        errors.append("Target dimensions must be positive integers")
    
    if not (0 < CRF_VALUE <= 51):
        errors.append("CRF_VALUE must be between 1 and 51")
    
    if MAX_VIDEO_DURATION <= MIN_VIDEO_DURATION:
        errors.append("MAX_VIDEO_DURATION must be greater than MIN_VIDEO_DURATION")
    
    return errors

def get_output_dir():
    """Get absolute path to output directory"""
    return Path(OUTPUT_DIR).absolute()

def get_temp_dir():
    """Get temporary directory path"""
    import tempfile
    return tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)