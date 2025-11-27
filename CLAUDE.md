# AI Reel Generation Agent - Complete Implementation Plan

---

## 🎯 PROGRESS - COMPLETE FLOW TILL HERE

### ✅ COMPLETED (100%)

**Phase 1: Foundation Setup** ✅

- [x] Project directory structure (`services/`, `workflows/`, `utils/`, `output/`)
- [x] `.env` configured with all API keys (Pexels, HeyGen, Submagic, Fal AI, OpenAI, Claude, Supabase)
- [x] `config.py` - Complete Config class with AI Agent options
- [x] `requirements.txt` - All dependencies including OpenAI, Anthropic, langdetect
- [x] `utils/file_uploader.py` - Production-ready Supabase storage integration
- [x] `.gitignore` created
- [x] Virtual environment configured

**AI Agent Service** ✅ NEW!

- [x] `services/ai_agent_service.py` - Two-stage AI decision engine
  - Stage 1: Claude API refines user's raw idea into creative brief
  - Stage 2: GPT-3.5-turbo makes structured decisions (category, keywords, music, voice, narration, emotion)
  - Language detection (Hindi/English/Arabic/Urdu)
  - JSON validation against Config options
- [x] Dependencies installed: `openai>=1.13.3`, `anthropic>=0.18.0`, `langdetect>=1.0.9`

**Main Entry Point** ✅ NEW!

- [x] `main.py` - New user flow implemented
  - Step 1: Get user's video idea (single input)
  - Step 2: Choose workflow (1. Local Media, 2. Pexels Stock, 3. HeyGen Avatar)
  - Clean, intuitive UI with emojis and examples

**Pexels Stock Workflow - AI-Driven** ✅

- [x] Update `workflows/pexels_workflow.py` to use AI Agent decisions
- [x] Integrate AI-generated decisions into video search
- [x] Update `services/audio_service.py` with voice_id, emotion, language parameters
- [x] Test end-to-end AI-driven Pexels workflow

### ✅ COMPLETED - Local Media Workflow

**Local Media Workflow** ✅

- [x] Create `workflows/local_media_workflow.py`
- [x] Implement directory scanning for video files
- [x] Implement dynamic video trimming (20s total / number of clips, e.g., 4 videos = 5s each, 5 videos = 4s each)
- [x] Implement video concatenation
- [x] Integrate Supabase upload
- [x] Integrate Submagic for subtitles/effects
- [x] Update `main.py` to include local media workflow option

### ✅ COMPLETED - HeyGen Avatar + B-Roll Workflow (Instagram Reels Style)

**HeyGen Workflow** ✅ **NEW! (Nov 26, 2025)**

- [x] Create `workflows/heygen_workflow.py` - AI-driven avatar + B-roll workflow
- [x] Implement AI Agent integration (script, keywords, emotion detection)
- [x] Implement B-roll preparation (6 clips × 4s = 24s, Pexels API)
- [x] Implement HeyGen avatar generation (transparent background, matting=true)
- [x] Implement FFmpeg compositing (avatar over B-roll, Instagram style)
- [x] Fix HeyGen emotion mapping (AI emotions → HeyGen emotions)
- [x] Fix HeyGen video status endpoint (v1 endpoint, not v2)
- [x] Fix HeyGen timeout (increased to 900s / 15 minutes)
- [x] Update `utils/video_processor.py` - composite_avatar_with_broll() for Instagram layout
- [x] Integrate Supabase upload for composited video
- [x] Integrate Submagic for subtitles/effects on final video
- [x] Update `main.py` to include HeyGen workflow option

**Core Services** ✅

- [x] `services/pexels_service.py` - Pexels API client
- [x] `services/heygen_service.py` - HeyGen API client (v2 generate, v1 status)
- [x] `services/submagic_service.py` - Submagic API client
- [x] `services/audio_service.py` - Fal AI integration (music + enhanced TTS)
- [x] `utils/video_processor.py` - FFmpeg wrapper (trim, concat, composite, chromakey)
- [x] `utils/file_uploader.py` - Supabase storage with signed URLs

### 🔜 PENDING (0%)

**Phase 4: Testing & Polish**

- [ ] Unit tests for all services
- [ ] Integration tests for workflows
- [ ] End-to-end testing with real APIs
- [ ] Performance optimization
- [ ] Error handling improvements

---

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
- **Production**: **Supabase Storage** with signed URLs (IMPLEMENTED ✅)

---

## Implementation Plan - By Workflow (NOT Generic Phases)

The implementation is organized by **3 complete workflows**, not abstract phases. Each workflow is built end-to-end before moving to the next.

### ✅ PHASE 1: FOUNDATION SETUP (COMPLETE)

**Duration**: Day 1
**Status**: ✅ **COMPLETE**

#### What's Done:

- [x] Project directory structure (`services/`, `workflows/`, `utils/`, `output/`)
- [x] `.env` configured with all API keys (Pexels, HeyGen, Submagic, Fal AI, Supabase)
- [x] `config.py` - Complete Config class with all settings
- [x] `requirements.txt` - All dependencies including Supabase
- [x] **`utils/file_uploader.py`** - Production-ready Supabase storage integration ✅
- [x] `.gitignore` created
- [x] Virtual environment configured
- [x] All foundation files in place and tested

---

### WORKFLOW 1: Pexels Stock Reel Generator (AI-DRIVEN) ⭐

**Duration**: Days 2-4
**Status**: 🔄 **IN PROGRESS** (Refactoring existing code)
**Goal**: AI-powered one-input video generation - user provides idea, AI agent handles everything

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
OPENAI_API_KEY=your_openai_key_here          # NEW - For AI Agent

# Supabase Storage (Production)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key_here
SUPABASE_BUCKET=reel-videos

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # NEW - For AI Agent

    # Supabase Storage
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reel-videos")

    # HeyGen Avatar IDs
    HEYGEN_AVATARS = [
        {"id": "Angela-inblackskirt-20220820", "name": "Angela (Professional Female)"},
        {"id": "josh-incasualtshirt-20220820", "name": "Josh (Casual Male)"},
        {"id": "monica-inpinkskirt-20220820", "name": "Monica (Business)"},
        {"id": "wayne-incasualsuit-20220820", "name": "Wayne (Corporate)"},
    ]

    # AI Agent Voice Options - NEW
    VOICE_IDS = [
        "Custom", "Wise_Woman", "Friendly_Person", "Inspirational_Girl",
        "Deep_Voice_Man", "Calm_Women", "Casual_Guy", "Lively_Girl",
        "Patient_Man", "Young_Knight", "Determined_Man", "Lovely_Girl", "Decent_Boy"
    ]

    EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
    LANGUAGES = ["English", "Hindi", "Arabic", "Urdu"]

    MUSIC_STYLES = [
        "Upbeat & Energetic", "Calm & Peaceful", "Cinematic & Epic",
        "Corporate & Professional", "Hip-Hop & Urban", "Pop & Catchy"
    ]

    VOICE_STYLES = [
        "Professional Narrator", "Friendly & Casual", "Energetic & Excited",
        "Calm & Soothing", "Authoritative"
    ]

    VIDEO_CATEGORIES = [
        "Nature & Lifestyle", "Urban & City Life", "People & Activities",
        "Abstract & Creative", "Seasonal & Weather"
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

# AI/ML - NEW for AI Agent
openai==1.12.0              # GPT-4 for decision engine
langdetect==1.0.9           # Language detection (Hindi/English)

# Video processing
ffmpeg-python==0.2.0

# Async support
aiohttp==3.9.1
asyncio==3.4.3

# File storage - Supabase
supabase==2.3.0
storage3==0.8.0

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

#### 2.1 Service Implementations (Code Templates in Codebase)

**`services/ai_agent_service.py`** ⭐ **NEW - AI Decision Engine**

- `analyze_idea(user_idea: str) -> dict` - OpenAI GPT-4 analyzes user's video idea
- Returns structured decisions:
  ```python
  {
    "category": str,           # Video category
    "keywords": [str],         # Pexels search keywords
    "duration": str,           # short/medium/long
    "music_style": str,        # Music genre/mood
    "voice_style": str,        # Narration tone
    "narration": str,          # Generated 20-second script
    "voice_id": str,           # Selected voice ID
    "emotion": str,            # Voice emotion
    "language": str            # Detected language (Hindi/English)
  }
  ```
- Uses GPT-4 with structured output prompting
- Language detection via langdetect library
- Script generation optimized for reel engagement

**`services/pexels_service.py`**

- `search_videos(query, per_page)` - Search Pexels API for portrait videos
- `download_video(url, path)` - Download video to local filesystem

**`services/heygen_service.py`**

- `generate_video(script, avatar_id)` - Generate 9:16 avatar video
- `wait_for_completion(video_id)` - Poll until video ready, return URL
- Payload: avatar config, voice (en-US-JennyNeural), white background

**`services/submagic_service.py`**

- `edit_video(video_url, enable_subtitles)` - Submit to Submagic API
- `wait_for_completion(job_id)` - Poll job status, return download URL
- Settings: captions, transitions, 9:16 aspect ratio

**`services/audio_service.py`**

- `generate_music(prompt, duration)` - Fal AI music generation
- `text_to_speech(text, voice, emotion, language)` - TTS with advanced options
- Enhanced parameters: voice_id, emotion, language support

**`utils/file_uploader.py`** (✅ IMPLEMENTED - Supabase)

- `upload_supabase(file_path)` - Upload to Supabase storage bucket
- Returns signed URL valid for 1 hour
- Production-ready with error handling

**`utils/video_processor.py`**

- `trim_video(path, duration)` - Trim to specified seconds
- `calculate_clip_duration(num_clips, target_total=20)` - Calculate per-clip duration (e.g., 4 clips = 5s, 5 clips = 4s)
- `concatenate_videos(paths, output)` - Stitch multiple videos
- `add_audio(video_path, audio_path)` - Mix video + audio track
- `get_video_info(path)` - Extract duration, dimensions, fps
- `composite_avatar_with_broll(broll, avatar, position)` - Picture-in-picture overlay (30% avatar size, 20px padding)
- Uses ffmpeg-python for all operations

---

### PHASE 3: Workflow Implementation

**Duration**: Days 6-9
**Goal**: Build end-to-end logic for all three user scenarios

#### 3.1 `workflows/local_media_workflow.py`

**Steps**: Scan directory → Trim videos (dynamic: 20s total / number of clips) → Concatenate → Generate audio (music + voice) → Mix → Upload to Supabase → Submit to Submagic (with subtitles) → Download final output

#### 3.2 `workflows/pexels_workflow.py` - AI-DRIVEN WORKFLOW ⭐

**NEW APPROACH**: One input from user → AI agent makes ALL decisions

**USER INPUT** (Single prompt):

```
"Tell me about your video idea:"
Example: "I want a calming video about morning coffee routines"
```

**AI AGENT PROCESSING** (OpenAI GPT-4):
The AI agent analyzes the user's idea and automatically determines:

1. **Video Category** (Auto-selected from keywords):

   - Nature & Lifestyle
   - Urban & City Life
   - People & Activities
   - Abstract & Creative
   - Seasonal & Weather
   - Custom keywords extracted from idea

2. **Video Duration** (Based on generated script length):

   - Short (5-15s)
   - Medium (15-30s)
   - Long (30+s)
   - AI generates script first, then calculates duration

3. **Audio Settings** (FIXED):

   - Always: Music + Voice Narration (Fal AI TTS)

4. **Music Style** (AI chooses based on idea context):

   - Upbeat & Energetic
   - Calm & Peaceful
   - Cinematic & Epic
   - Corporate & Professional
   - Hip-Hop & Urban
   - Pop & Catchy

5. **Voice Style** (AI chooses based on tone):

   - Professional Narrator
   - Friendly & Casual
   - Energetic & Excited
   - Calm & Soothing
   - Authoritative

6. **Narration Script** (AI generates ~20 seconds):

   - Auto-generated based on video idea
   - Optimized for reel engagement
   - Target: 20-second duration

7. **Voice ID** (AI selects from available voices):

   - Custom, Wise_Woman, Friendly_Person, Inspirational_Girl
   - Deep_Voice_Man, Calm_Women, Casual_Guy, Lively_Girl
   - Patient_Man, Young_Knight, Determined_Man, Lovely_Girl, Decent_Boy

8. **Emotion** (AI chooses based on video mood):

   - happy, sad, angry, fearful, surprised, disgusted, neutral

9. **Language** (Auto-detected from user input):
   - If user writes in Hindi → Hindi
   - Else → English (default)

**WORKFLOW EXECUTION**:

```
User Idea → AI Agent (OpenAI) → Decision JSON → Pexels Search → Video Download →
Audio Generation (Fal AI Music + Voice) → Video + Audio Mixing → Upload (Supabase) →
Submagic Editing (WITH SUBTITLES + Effects) → Final Download
```

**IMPORTANT - Subtitle Handling**:

- ❌ **Remove Fal AI subtitle generation** (if currently used)
- ✅ **Use Submagic for subtitles** (auto-generated from voice narration)
- Submagic will analyze the audio track and generate captions automatically
- `enable_subtitles=True` when calling `submagic.edit_video()`

**IMPLEMENTATION**:

```python
class PexelsWorkflow:
    def execute(self, user_idea: str, output_path: str):
        # Step 1: AI Agent makes all decisions
        decisions = self.ai_agent.analyze_idea(user_idea)
        # Returns: {
        #   "category": "Nature & Lifestyle",
        #   "keywords": ["morning coffee", "sunrise", "peaceful routine"],
        #   "duration": "medium",
        #   "music_style": "Calm & Peaceful",
        #   "voice_style": "Calm & Soothing",
        #   "narration": "Wake up to the perfect morning...",
        #   "voice_id": "Calm_Women",
        #   "emotion": "happy",
        #   "language": "English"
        # }

        # Step 2: Search Pexels with AI-extracted keywords
        videos = self.pexels.search_videos(decisions["keywords"])

        # Step 3: Download, trim (6s each), concatenate
        # Step 4: Generate music based on AI-chosen style (Fal AI)
        music_url = self.audio.generate_music(
            prompt=decisions["music_style"],
            duration=30
        )

        # Step 5: Generate voice narration (Fal AI TTS)
        voice_url = self.audio.text_to_speech(
            text=decisions["narration"],
            voice=decisions["voice_id"],
            emotion=decisions["emotion"],
            language=decisions["language"]
        )

        # Step 6: Mix video + music + voice
        final_video = self.processor.mix_audio_tracks(
            video_path, music_path, voice_path
        )

        # Step 7: Upload to Supabase
        public_url = self.uploader.upload_supabase(final_video)

        # Step 8: Submagic - Add subtitles + effects
        # Submagic auto-generates captions from the voice narration audio
        job = self.submagic.edit_video(
            public_url,
            enable_subtitles=True  # ✅ Submagic generates subtitles
        )

        # Step 9: Download final video with subtitles
        download_url = self.submagic.wait_for_completion(job["id"])
        self.submagic.download_video(download_url, output_path)
```

**KEY DIFFERENCE**:

- User provides 1 input (creative idea)
- AI handles 100% of technical decisions
- Subtitles generated by Submagic (NOT Fal AI)

---

### AI AGENT PROMPT ENGINEERING (GPT-4)

**System Prompt for `ai_agent_service.py`**:

```
You are a professional video production AI assistant. Analyze the user's video idea and make optimal decisions for creating an engaging social media reel.

Given a user's video idea, return a JSON object with these fields:
- category: Select from [Nature & Lifestyle, Urban & City Life, People & Activities, Abstract & Creative, Seasonal & Weather]
- keywords: Array of 3-5 Pexels search keywords
- duration: "short" (5-15s), "medium" (15-30s), or "long" (30+s)
- music_style: Select from [Upbeat & Energetic, Calm & Peaceful, Cinematic & Epic, Corporate & Professional, Hip-Hop & Urban, Pop & Catchy]
- voice_style: Select from [Professional Narrator, Friendly & Casual, Energetic & Excited, Calm & Soothing, Authoritative]
- narration: Generate a compelling 20-second script for TTS (approximately 50-60 words)
- voice_id: Select from [Wise_Woman, Friendly_Person, Inspirational_Girl, Deep_Voice_Man, Calm_Women, Casual_Guy, Lively_Girl, Patient_Man, Young_Knight, Determined_Man, Lovely_Girl, Decent_Boy]
- emotion: Select from [happy, sad, angry, fearful, surprised, disgusted, neutral]

The narration should be engaging, concise, and optimized for social media virality.
Match the tone to the video concept.
```

**Example AI Response**:

```json
{
  "category": "Nature & Lifestyle",
  "keywords": [
    "morning coffee",
    "sunrise",
    "peaceful routine",
    "kitchen",
    "lifestyle"
  ],
  "duration": "medium",
  "music_style": "Calm & Peaceful",
  "voice_style": "Calm & Soothing",
  "narration": "Wake up to the perfect morning ritual. There's something magical about that first sip of coffee as the sun rises. The warmth in your hands, the rich aroma filling the air. This is your moment of peace before the day begins.",
  "voice_id": "Calm_Women",
  "emotion": "happy",
  "language": "English"
}
```

**Language Detection**:

- Use `langdetect` library to detect if input contains Hindi/Urdu scripts
- If detected: set `language: "Hindi"` and ensure narration is in Hindi
- Default: `language: "English"`

---

#### 3.3 `workflows/heygen_workflow.py` ⭐ **PRODUCTION FLOW - HEYGEN + SUBMAGIC ONLY**

**CURRENT IMPLEMENTATION** (Updated: Nov 26, 2025 - Verified from codebase):

**Philosophy**: Let HeyGen generate full-screen avatar, let Submagic add AI B-rolls + subtitles. NO Pexels, NO FFmpeg compositing!

---

### **WORKFLOW STEPS** (As Implemented in Code):

#### **Step 1: User Input**
- User provides video idea (single text input)
- User selects avatar from interactive menu (`display_avatar_menu()`)
- Shows avatar names with gender icons (👨/👩)
- Returns selected avatar dict with `id`, `name`, `gender`

#### **Step 2: HeyGen Avatar Generation** (Full Screen, White Background)
**Code**: `heygen_workflow.py:242-250`

```python
heygen_result = self.heygen.generate_video(
    script=script,
    avatar_id=avatar_id,
    voice_id=voice_id,
    background_type="color",
    background_value="#FFFFFF",  # White background
    title=title or "AI Avatar Reel",
    emotion=emotion
)
```

**HeyGen Service Configuration** (`heygen_service.py:81-93`):
- `scale`: 1.0 (full-screen avatar, not scaled down)
- `offset`: {"x": 0, "y": 0} (centered horizontally and vertically)
- `matting`: false (solid background, no transparency)
- `background_type`: "color"
- `background_value`: "#FFFFFF" (white background)
- `super_resolution`: true (high quality)
- `talking_style`: "stable" (consistent head movement)

**Emotion Mapping** (`heygen_service.py:63-73`):
- AI emotions (happy, sad, calm, etc.) are automatically mapped to HeyGen-compatible emotions
- HeyGen accepts: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster', 'Angry'
- Mapping defined in `Config.HEYGEN_EMOTION_MAPPING`
- Unknown emotions fallback to 'Friendly'

**Polling Strategy** (`heygen_service.py:202-264`):
- **NO TIMEOUT** - waits indefinitely until video completes
- Poll interval: 10 seconds
- Accepts statuses: "processing", "pending", "queued", "completed"
- Fails on: "failed", "error"
- Uses **v1 endpoint** for status checks: `https://api.heygen.com/v1/video_status.get`

#### **Step 3: Download HeyGen Video**
**Code**: `heygen_workflow.py:260-274`
- Downloads HeyGen video to temp directory
- Saves copy to output folder: `output/heygen_raw_YYYYMMDD_HHMMSS.mp4`
- This is the **raw avatar video** before Submagic processing

#### **Step 4: Upload to Supabase**
**Code**: `heygen_workflow.py:276-278`
- Uploads HeyGen video to Supabase storage
- Gets public URL for Submagic API consumption
- Uses `FileUploader.upload()` method (returns signed URL)

#### **Step 5: Submagic Processing** (Magic B-rolls + Subtitles)
**Code**: `heygen_workflow.py:280-296`

```python
final_path = self.submagic.process_video(
    video_url=heygen_supabase_url,
    output_path=str(output_path),
    title="AI Generated Avatar Reel",
    language="en",
    template_name="Alex",  # Subtitle positioning template
    magic_zooms=False,     # No auto-zoom effects
    magic_brolls=True,     # AI B-rolls from Submagic library
    magic_brolls_percentage=50  # 50% B-roll coverage
)
```

**Submagic Service Workflow** (`submagic_service.py:359-479`):
1. **Create Project** - Submit video URL, returns `project_id`
2. **Wait for Processing** - Poll status until "ready" or "completed"
3. **Export Video** - Trigger final rendering, get download URL
4. **Download** - Save edited video to local filesystem

**Submagic Features Applied**:
- **Auto-generated subtitles** from avatar speech (98%+ accuracy)
- **Magic B-rolls** (50% coverage) - AI selects relevant stock footage
- **Alex template** - Professional caption styling with specific positioning
- **NO Magic Zooms** - Disabled for cleaner output
- **Language**: English (default, can be made dynamic)

**Timeout**: 600 seconds (10 minutes) for Submagic processing

#### **Step 6: Download Final Video**
**Code**: `heygen_workflow.py:298-307`
- Downloads final video from Submagic
- Saved to: output path specified by user
- Typically: `output/final_with_subtitles_YYYYMMDD_HHMMSS.mp4`

---

### **ERROR HANDLING**:

**Submagic Failure Fallback** (`heygen_workflow.py:309-320`):
- If Submagic fails, workflow returns the raw HeyGen video
- User gets `heygen_raw_YYYYMMDD_HHMMSS.mp4` (avatar without B-rolls/subtitles)
- Prints warning but doesn't crash the entire workflow

---

### **WHAT WAS REMOVED** (From Old Flows):

❌ **Pexels API Integration**:
- Old code had `prepare_broll()` method (still exists but unused in current flow)
- Old code had `extract_keywords_from_script()` (still exists but unused)
- No B-roll fetching from Pexels anymore

❌ **FFmpeg Compositing**:
- No manual video compositing
- No `composite_avatar_with_broll()` usage
- No chromakey/green screen removal

❌ **Complex Positioning**:
- No custom avatar scaling/positioning logic
- No Instagram-style bottom-half layout
- Simple full-screen centered approach

---

### **OUTPUT FILES**:

1. **`output/heygen_raw_YYYYMMDD_HHMMSS.mp4`**
   - Raw HeyGen avatar video
   - Full-screen avatar with white background
   - No subtitles, no B-rolls
   - Saved before Submagic processing

2. **`output/final_with_subtitles_YYYYMMDD_HHMMSS.mp4`** (or user-specified path)
   - Final edited reel from Submagic
   - Includes AI-generated B-rolls (50% coverage)
   - Includes auto-generated subtitles (Alex template)
   - Production-ready for social media

---

### **KEY TECHNICAL DETAILS**:

**HeyGen API**:
- Endpoint: `POST https://api.heygen.com/v2/video/generate`
- Status check: `GET https://api.heygen.com/v1/video_status.get?video_id={id}` (v1, not v2!)
- Dimensions: 720x1280 (9:16 portrait)
- Infinite polling until completion

**Submagic API**:
- Create: `POST {base_url}/projects`
- Status: `GET {base_url}/projects/{id}`
- Export: `POST {base_url}/projects/{id}/export`
- Header: `x-api-key` (NOT Authorization Bearer)
- Timeout: 600s (configurable in Config)

**Supabase Storage**:
- Upload via `FileUploader.upload()` method
- Returns signed URL valid for 1 hour
- Required for Submagic API (needs public URL)

---

### **UNUSED METHODS** (Still in Code but Not Called):

- `extract_keywords_from_script()` - Lines 81-121 (for Pexels, not used)
- `prepare_broll()` - Lines 123-205 (Pexels B-roll preparation, not used)
- `generate_video_with_broll()` in `heygen_service.py` (Lines 306-337) - Convenience method for video backgrounds

These methods remain in the codebase for potential future use but are **not part of the current production flow**.

---

### PHASE 4: Integration & Main Dispatcher

**Duration**: Days 10-11
**Goal**: Build the central dispatcher and integrate all workflows

#### 4.1 `main.py` - Central Dispatcher

**Functions**:

- `display_menu()` - Show 3 workflow options + exit
- `create_output_path(workflow_name)` - Generate timestamped filenames
- `run_local_media_workflow()` - Prompt for directory, execute LocalMediaWorkflow
- `run_pexels_workflow()` - **AI-DRIVEN**: Single prompt "Tell me about your video idea:", AI handles everything
- `run_heygen_workflow()` - Prompt for avatar + script + B-roll topic, execute HeyGenWorkflow
- `main()` - Loop menu until user exits

**Entry Point**: Validates API keys (including OPENAI_API_KEY) in Config before starting menu loop

**NEW Pexels Workflow Menu**:

```
🎬 AI REEL GENERATOR - PERSONALIZED VIDEO CREATOR
================================================================================
📱 Create perfect Instagram Reels, TikTok videos, and YouTube Shorts
🎯 Just tell us your idea - AI handles everything!
================================================================================

💡 Tell me about your video idea:
Example: "I want a calming video about morning coffee routines"
Example: "मुझे सुबह की योग प्रैक्टिस के बारे में एक वीडियो चाहिए"

Your idea: _____

⚡ AI will automatically:
   ✓ Choose video category & search keywords
   ✓ Determine optimal duration
   ✓ Select music style
   ✓ Pick voice style & emotion
   ✓ Generate 20-second narration script
   ✓ Detect language (Hindi/English)
   ✓ Create professional reel

🚀 Press Enter to generate...
```

---

## Testing Strategy

**Unit Tests** (`tests/test_services.py`): Mock API responses for PexelsService, HeyGenService, SubmagicService
**Integration Tests** (`tests/test_workflows.py`): Test workflow execution with dummy files
**Run**: `python -m unittest discover tests/`

---

## Deployment Checklist

**Pre-Deployment**: API keys in `.env` • FFmpeg installed • Python 3.12.7 • venv activated • Dependencies installed
**Testing**: Validate all API keys • Unit + integration tests pass • Test each workflow end-to-end
**Production**: ✅ Supabase storage configured • Error logging • Rate limiting • Retry logic • Monitoring

## Supabase Storage Setup ✅

1. Install: `pip install supabase==2.3.0 storage3==0.8.0`
2. Create project at https://supabase.com → Get URL + Anon Key
3. Create bucket: `reel-videos` (Private)
4. Add to `.env`: `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_BUCKET`

## Troubleshooting

| Issue             | Solution                                                            |
| ----------------- | ------------------------------------------------------------------- |
| FFmpeg not found  | `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Ubuntu)       |
| HeyGen timeout    | Increase timeout in `wait_for_completion()`                         |
| Submagic fails    | Verify video URL is publicly accessible (check Supabase signed URL) |
| API rate limiting | Implement exponential backoff, check plan limits                    |

## Future Enhancements

**Web Interface**: Flask/FastAPI dashboard, drag-and-drop upload, real-time progress
**Batch Processing**: Multi-topic queue system
**AI Features**: GPT script generation, content moderation, topic suggestions
**Multi-Platform**: Instagram Reels, TikTok, YouTube Shorts optimized exports

---

## Project Timeline

| Phase                      | Duration   | Status          | Deliverables                                                                                              |
| -------------------------- | ---------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| **Phase 1: Foundation**    | Day 1      | ✅ **COMPLETE** | Project structure, config.py, .env, requirements.txt, file_uploader.py (Supabase), .gitignore, venv setup |
| **Phase 2: Core Services** | Days 2-4   | 🔜 NEXT         | pexels_service, heygen_service, submagic_service, audio_service, video_processor                          |
| **Phase 3: Workflows**     | Days 5-8   | 🔜 PENDING      | local_media_workflow, pexels_workflow, heygen_workflow                                                    |
| **Phase 4: Integration**   | Days 9-10  | 🔜 PENDING      | main.py dispatcher, menu system                                                                           |
| **Phase 5: Testing**       | Days 11-12 | 🔜 PENDING      | Unit tests, integration tests, end-to-end validation                                                      |

**Total**: 12 days

---

## Summary

**Architecture**: 3-workflow dispatcher system with AI-powered decision engine
**Current Status**: ✅ Phase 1 Complete - Foundation ready
**Next Steps**: Implement AI Agent Service + Core Services (Pexels, HeyGen, Submagic, Audio, VideoProcessor)

**Key Features**:

- **Local Media**: Process existing videos with Submagic editing
- **Pexels Stock** ⭐ **AI-DRIVEN**: Single user input → AI decides everything (category, music, voice, script, emotion)
- **HeyGen Avatar**: YouTube-style videos with AI avatar + B-roll compositing

**Tech Stack**:

- Python 3.12.7 • FFmpeg • Supabase Storage
- APIs: Pexels • HeyGen • Submagic • Fal AI • **OpenAI GPT-4** (NEW)
- AI: Language detection (Hindi/English) • Automated script generation • Voice/music selection

**Innovation**: Users describe their vision in natural language, AI handles 100% of technical production decisions
