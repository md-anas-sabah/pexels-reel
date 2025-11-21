"""
Configuration settings for AI Reel Generation Agent
Supports three workflows: Local Media, Pexels Stock, HeyGen Avatar + B-Roll
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for all API keys and settings"""

    # ==================== API KEYS ====================

    # Pexels API (Video Search & Download)
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    PEXELS_BASE_URL = "https://api.pexels.com/videos"

    # HeyGen API (AI Avatar Generation)
    HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
    HEYGEN_BASE_URL = "https://api.heygen.com/v2"

    # Submagic API (Automated Video Editing)
    SUBMAGIC_API_KEY = os.getenv("SUBMAGIC_API_KEY")
    SUBMAGIC_BASE_URL = "https://api.submagic.co/v1"

    # Fal AI API (Audio Generation - Music & TTS)
    FAL_AI_API_KEY = os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")

    # OpenAI API (AI Agent - GPT-3.5-turbo for decisions)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Anthropic API (AI Agent - Claude for prompt refinement)
    ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")  # Maps from CLAUDE_API_KEY in .env

    # Supabase (File Storage for Public URLs)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reel-videos")

    # ==================== HEYGEN AVATARS ====================

    HEYGEN_AVATARS = [
        {"id": "Angela-inblackskirt-20220820", "name": "Angela (Professional Female)"},
        {"id": "josh-incasualtshirt-20220820", "name": "Josh (Casual Male)"},
        {"id": "monica-inpinkskirt-20220820", "name": "Monica (Business)"},
        {"id": "wayne-incasualsuit-20220820", "name": "Wayne (Corporate)"},
    ]

    # ==================== AI AGENT OPTIONS ====================

    # Voice IDs for TTS (Fal AI)
    VOICE_IDS = [
        "Wise_Woman", "Friendly_Person", "Inspirational_Girl",
        "Deep_Voice_Man", "Calm_Women", "Casual_Guy", "Lively_Girl",
        "Patient_Man", "Young_Knight", "Determined_Man", "Lovely_Girl", "Decent_Boy"
    ]

    # Voice ID Mapping - Maps our custom IDs to actual MiniMax API voice IDs
    # Only "Wise_Woman" is confirmed to work in MiniMax API
    VOICE_ID_API_MAPPING = {
        "Calm_Women": "Wise_Woman",
        "Friendly_Person": "Wise_Woman",
        "Inspirational_Girl": "Wise_Woman",
        "Deep_Voice_Man": "Wise_Woman",
        "Casual_Guy": "Wise_Woman",
        "Lively_Girl": "Wise_Woman",
        "Patient_Man": "Wise_Woman",
        "Young_Knight": "Wise_Woman",
        "Determined_Man": "Wise_Woman",
        "Lovely_Girl": "Wise_Woman",
        "Decent_Boy": "Wise_Woman",
        "Wise_Woman": "Wise_Woman"
    }

    # Voice Emotions
    EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]

    # Supported Languages
    LANGUAGES = ["English", "Hindi", "Arabic", "Urdu"]

    # Music Styles
    MUSIC_STYLES = [
        "Upbeat & Energetic", "Calm & Peaceful", "Cinematic & Epic",
        "Corporate & Professional", "Hip-Hop & Urban", "Pop & Catchy"
    ]

    # Voice Styles
    VOICE_STYLES = [
        "Professional Narrator", "Friendly & Casual", "Energetic & Excited",
        "Calm & Soothing", "Authoritative"
    ]

    # Video Categories
    VIDEO_CATEGORIES = [
        "Nature & Lifestyle", "Urban & City Life", "People & Activities",
        "Abstract & Creative", "Seasonal & Weather"
    ]

    # ==================== VIDEO PROCESSING SETTINGS ====================

    # Target dimensions for social media reels (9:16 aspect ratio)
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280
    TARGET_ASPECT_RATIO = 9.0 / 16.0

    # Quality Settings (High Quality)
    VIDEO_CODEC = "libx264"
    AUDIO_CODEC = "aac"
    CRF_VALUE = 18  # Lower = better quality (18 is visually lossless)
    PRESET = "slower"  # Better quality preset
    PIXEL_FORMAT = "yuv420p"
    VIDEO_PROFILE = "high"
    VIDEO_LEVEL = "4.0"
    AUDIO_BITRATE = "128k"

    # Object Detection Settings
    FACE_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 5
    FACE_MIN_SIZE = (30, 30)

    BODY_SCALE_FACTOR = 1.1
    BODY_MIN_NEIGHBORS = 3
    BODY_MIN_SIZE = (50, 50)

    # ROI (Region of Interest) Settings
    ROI_PADDING_FACTOR = 0.2  # 20% padding around detected objects
    SAMPLE_FRAME_INTERVAL = 20  # Analyze every 20th frame

    # ==================== OUTPUT SETTINGS ====================

    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
    TEMP_DIR_PREFIX = "video_reel_temp_"

    # ==================== FILE SETTINGS ====================

    MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 60))
    MIN_VIDEO_DURATION = 5
    PREFERRED_QUALITY = ["hd", "sd"]
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv"]
    ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    MAX_FILE_SIZE_MB = 500

    # ==================== SUBMAGIC SETTINGS ====================

    SUBMAGIC_ENABLE_SUBTITLES = True  # Default: enable subtitles
    SUBMAGIC_CAPTION_STYLE = "modern"
    SUBMAGIC_ADD_TRANSITIONS = True
    SUBMAGIC_ADD_MUSIC = False  # We handle music separately
    SUBMAGIC_ASPECT_RATIO = "9:16"

    # ==================== WORKFLOW SETTINGS ====================

    # Multi-clip reel settings
    DEFAULT_CLIP_COUNT = 6
    DEFAULT_SEGMENT_DURATION = 3.5  # seconds per segment
    MIN_CLIPS = 2
    MAX_CLIPS = 15

    # Logo overlay settings
    LOGO_WIDTH = 80  # pixels
    LOGO_POSITION_X = 15  # pixels from right edge
    LOGO_POSITION_Y = 25  # pixels from top

    # ==================== CREWAAI SETTINGS ====================

    CREW_VERBOSE = True
    CREW_PROCESS = "sequential"

    # ==================== LOGGING CONFIGURATION ====================

    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    # ==================== ERROR HANDLING ====================

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    # ==================== PERFORMANCE SETTINGS ====================

    MAX_CONCURRENT_DOWNLOADS = 3
    CHUNK_SIZE = 8192  # bytes for video download

    # ==================== TIMEOUT SETTINGS ====================

    HEYGEN_TIMEOUT = 300  # 5 minutes for avatar generation
    SUBMAGIC_TIMEOUT = 600  # 10 minutes for video editing

    # ==================== SUPABASE STORAGE SETTINGS ====================

    SUPABASE_URL_EXPIRY = 3600  # 1 hour for signed URLs
    SUPABASE_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max upload


# ==================== VALIDATION FUNCTIONS ====================

def validate_config():
    """Validate all configuration settings"""
    errors = []
    warnings = []

    # Check required API keys
    if not Config.PEXELS_API_KEY:
        errors.append("PEXELS_API_KEY is required")

    if not Config.HEYGEN_API_KEY:
        warnings.append("HEYGEN_API_KEY not set (HeyGen workflow will fail)")

    if not Config.SUBMAGIC_API_KEY:
        warnings.append("SUBMAGIC_API_KEY not set (Submagic editing will fail)")

    if not Config.FAL_AI_API_KEY:
        warnings.append("FAL_AI_API_KEY not set (Audio generation will fail)")

    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        warnings.append("Supabase credentials not set (File upload will use temporary hosting)")

    # Validate dimensions
    if Config.TARGET_WIDTH <= 0 or Config.TARGET_HEIGHT <= 0:
        errors.append("Target dimensions must be positive integers")

    # Validate quality settings
    if not (0 < Config.CRF_VALUE <= 51):
        errors.append("CRF_VALUE must be between 1 and 51")

    # Validate duration settings
    if Config.MAX_VIDEO_DURATION <= Config.MIN_VIDEO_DURATION:
        errors.append("MAX_VIDEO_DURATION must be greater than MIN_VIDEO_DURATION")

    return errors, warnings


def get_output_dir():
    """Get absolute path to output directory"""
    output_path = Path(Config.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    return output_path.absolute()


def get_temp_dir():
    """Get temporary directory path"""
    import tempfile
    return tempfile.mkdtemp(prefix=Config.TEMP_DIR_PREFIX)


def print_config_status():
    """Print configuration status for debugging"""
    errors, warnings = validate_config()

    print("\n" + "="*60)
    print("AI REEL GENERATION AGENT - CONFIGURATION STATUS")
    print("="*60)

    print("\nAPI Keys Status:")
    print(f"  ✓ Pexels API:    {'✓ SET' if Config.PEXELS_API_KEY else '✗ MISSING'}")
    print(f"  ✓ HeyGen API:    {'✓ SET' if Config.HEYGEN_API_KEY else '✗ MISSING'}")
    print(f"  ✓ Submagic API:  {'✓ SET' if Config.SUBMAGIC_API_KEY else '✗ MISSING'}")
    print(f"  ✓ Fal AI API:    {'✓ SET' if Config.FAL_AI_API_KEY else '✗ MISSING'}")
    print(f"  ✓ Supabase:      {'✓ SET' if Config.SUPABASE_URL and Config.SUPABASE_KEY else '✗ MISSING'}")

    print(f"\nOutput Directory: {get_output_dir()}")
    print(f"Target Resolution: {Config.TARGET_WIDTH}x{Config.TARGET_HEIGHT} (9:16)")
    print(f"Video Quality: CRF {Config.CRF_VALUE}, Preset: {Config.PRESET}")

    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  • {error}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")

    if not errors:
        print("\n✅ Configuration is valid!")

    print("="*60 + "\n")

    return len(errors) == 0


# Legacy compatibility (maintain old config variables)
PEXELS_API_KEY = Config.PEXELS_API_KEY
PEXELS_BASE_URL = Config.PEXELS_BASE_URL
TARGET_WIDTH = Config.TARGET_WIDTH
TARGET_HEIGHT = Config.TARGET_HEIGHT
TARGET_ASPECT_RATIO = Config.TARGET_ASPECT_RATIO
OUTPUT_DIR = Config.OUTPUT_DIR
MAX_VIDEO_DURATION = Config.MAX_VIDEO_DURATION
