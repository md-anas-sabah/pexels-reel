# AI Reel Generation Agent - Implementation Plan

## Project Overview
This document outlines the complete implementation plan for building an AI Reel Generation Agent using Python, CrewAI, HeyGen API, and Submagic API. The system will automatically generate social media reels with intelligent decision-making capabilities.

---

## Architecture Overview

### Tech Stack
- **Language**: Python 3.12.7
- **Framework**: CrewAI (for multi-agent orchestration)
- **APIs**: HeyGen (avatar videos), Submagic (video editing & stock content)
- **Dependencies**: asyncio/multiprocessing for parallel processing

### Project Structure
```
reel_generator/
├── main.py                          # Entry point & orchestration
├── requirements.txt                 # Dependencies
├── .env                            # Environment variables (API keys)
├── README.md                       # Documentation
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration & constants
├── data/
│   ├── __init__.py
│   └── avatars.py                  # Avatar catalog
├── utils/
│   ├── __init__.py
│   ├── media_scanner.py            # Video file scanning
│   └── file_handler.py             # Output file management
├── core/
│   ├── __init__.py
│   └── decision_logic.py           # Business logic & reel type selection
├── agents/
│   ├── __init__.py
│   ├── media_analyzer_agent.py     # Analyzes input videos
│   ├── strategy_agent.py           # Determines reel strategy
│   ├── script_generation_agent.py  # Generates scripts for avatars
│   ├── avatar_selector_agent.py    # Manages avatar selection
│   ├── pipeline_execution_agent.py # Executes API workflows
│   └── quality_check_agent.py      # Post-generation validation
├── services/
│   ├── __init__.py
│   ├── heygen_service.py           # HeyGen API integration
│   └── submagic_service.py         # Submagic API integration
└── output/                         # Generated reels storage
    └── [timestamp_folders]/
        ├── final.mp4
        ├── subtitles.srt
        ├── thumbnail.jpg
        └── metadata.json
```

---

## Implementation Phases

### PHASE 1: Foundation & Core Logic (Days 1-2)

#### 1.1 Project Initialization
**Tasks:**
- Create project directory structure
- Initialize Git repository
- Set up Python virtual environment
- Create `.gitignore` (exclude `.env`, `output/`, `__pycache__/`)

**Commands:**
```bash
mkdir -p reel_generator/{config,data,utils,core,agents,services,output}
cd reel_generator
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### 1.2 Configuration Module (`config/settings.py`)
**Implementation:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
    SUBMAGIC_API_KEY = os.getenv("SUBMAGIC_API_KEY")

    # Credit Costs
    CREDIT_COSTS = {
        "avatar_reel": 2,
        "edited_reel": 1,
        "stock_reel": 1
    }

    # Processing Limits
    MAX_PARALLEL_JOBS = 5
    BULK_MODE_AVATAR_RATIO = 0.5  # 50% avatars in bulk mode

    # File Paths
    OUTPUT_DIR = "output"
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv"]
```

#### 1.3 Media Scanner (`utils/media_scanner.py`)
**Features:**
- Scan directories for video files
- Validate file formats
- Return video count and file paths

**Implementation:**
```python
import os
from pathlib import Path
from typing import List, Tuple
from config.settings import Settings

def scan_video_folder(path: str) -> Tuple[int, List[str]]:
    """
    Scans a folder for video files.

    Returns:
        Tuple of (count, list of file paths)
    """
    video_files = []
    path_obj = Path(path)

    if not path_obj.exists() or not path_obj.is_dir():
        raise ValueError(f"Invalid directory: {path}")

    for file in path_obj.rglob("*"):
        if file.suffix.lower() in Settings.SUPPORTED_VIDEO_FORMATS:
            video_files.append(str(file))

    return len(video_files), video_files
```

#### 1.4 Decision Logic (`core/decision_logic.py`)
**Business Rules:**
- Single video (1) → Avatar OR Edited Reel
- Multiple videos (2-5) → Avatar OR Edited Reel
- Bulk mode (6+) → 50% Avatar, 50% Edited/Stock

**Implementation:**
```python
from typing import Dict, List

def get_reel_options(video_count: int) -> Dict[str, any]:
    """
    Determines available reel options based on video count.

    Returns:
        Dictionary with available options and costs
    """
    if video_count == 0:
        return {
            "options": ["stock_reel"],
            "mode": "single",
            "credits_required": 1
        }
    elif 1 <= video_count <= 5:
        return {
            "options": ["avatar_reel", "edited_reel"],
            "mode": "single",
            "credits_required": None  # Depends on user choice
        }
    else:  # 6+
        return {
            "options": ["bulk_processing"],
            "mode": "bulk",
            "avatar_count": video_count // 2,
            "edited_stock_count": video_count - (video_count // 2),
            "credits_required": (video_count // 2) * 2 + (video_count - video_count // 2)
        }
```

#### 1.5 Avatar Data (`data/avatars.py`)
**Structure:**
```python
AVATARS = [
    {
        "id": "avatar_001",
        "name": "Professional Female",
        "category": "business",
        "thumbnail_url": "https://...",
        "voice_id": "voice_001"
    },
    {
        "id": "avatar_002",
        "name": "Casual Male",
        "category": "lifestyle",
        "thumbnail_url": "https://...",
        "voice_id": "voice_002"
    },
    # Add more avatars...
]

def get_avatar_by_id(avatar_id: str):
    return next((a for a in AVATARS if a["id"] == avatar_id), None)
```

#### 1.6 Dependencies (`requirements.txt`)
```txt
crewai>=0.1.0
python-dotenv>=1.0.0
requests>=2.31.0
aiohttp>=3.9.0
pydantic>=2.0.0
```

---

### PHASE 2: CrewAI Agent Foundation (Days 3-4)

#### 2.1 Agent Base Structure
**Common Pattern:**
```python
from crewai import Agent, Task
from typing import List

class BaseReelAgent(Agent):
    def __init__(self, role: str, goal: str, backstory: str):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=False
        )
```

#### 2.2 Media Analyzer Agent (`agents/media_analyzer_agent.py`)
**Responsibilities:**
- Analyze video content (duration, quality, format)
- Extract metadata
- Assess suitability for different reel types

**Implementation:**
```python
class MediaAnalyzerAgent(BaseReelAgent):
    def __init__(self):
        super().__init__(
            role="Video Content Analyzer",
            goal="Analyze video files to determine optimal reel generation strategy",
            backstory="Expert in video content analysis with deep understanding of social media trends"
        )

    def analyze_videos(self, video_paths: List[str]) -> dict:
        # Implement video analysis logic
        pass
```

#### 2.3 Strategy Agent (`agents/strategy_agent.py`)
**Responsibilities:**
- Decide between Avatar/Edited/Stock reels
- Optimize credit usage
- Apply business logic

#### 2.4 Script Generation Agent (`agents/script_generation_agent.py`)
**Responsibilities:**
- Generate engaging scripts for avatar reels
- Adapt tone based on content type
- Optimize for platform algorithms

---

### PHASE 3: Avatar Selection System (Days 5-6)

#### 3.1 Avatar Selector Agent (`agents/avatar_selector_agent.py`)
**Features:**
- Present avatar options to user
- Filter by category/style
- Store user preferences

**Implementation:**
```python
class AvatarSelectorAgent(BaseReelAgent):
    def __init__(self):
        super().__init__(
            role="Avatar Selection Specialist",
            goal="Help users select the perfect avatar for their reel",
            backstory="Expert in matching avatar personas with content types"
        )

    def present_avatars(self) -> str:
        from data.avatars import AVATARS

        print("\n=== Available Avatars ===")
        for i, avatar in enumerate(AVATARS, 1):
            print(f"{i}. {avatar['name']} ({avatar['category']})")

        choice = int(input("\nSelect avatar number: ")) - 1
        return AVATARS[choice]["id"]
```

#### 3.2 User Interaction Flow
- Display avatar thumbnails/descriptions
- Allow filtering by category
- Validate selection
- Store for reuse in bulk mode

---

### PHASE 4: API Integration (Days 7-10)

#### 4.1 HeyGen Service (`services/heygen_service.py`)
**Endpoints:**
- Create avatar video
- Generate voice-over
- Check generation status
- Download completed video

**Implementation:**
```python
import aiohttp
from config.settings import Settings

class HeyGenService:
    BASE_URL = "https://api.heygen.com/v1"

    def __init__(self):
        self.api_key = Settings.HEYGEN_API_KEY

    async def create_avatar_video(self, avatar_id: str, script: str) -> dict:
        """
        Creates an avatar video with the given script.

        Returns:
            Job ID and status
        """
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "avatar_id": avatar_id,
                "script": script,
                "voice_settings": {
                    "speed": 1.0,
                    "emotion": "neutral"
                }
            }

            async with session.post(
                f"{self.BASE_URL}/video/generate",
                headers=headers,
                json=payload
            ) as response:
                return await response.json()

    async def get_video_status(self, job_id: str) -> dict:
        # Poll for completion
        pass

    async def download_video(self, video_url: str, output_path: str):
        # Download completed video
        pass
```

#### 4.2 Submagic Service (`services/submagic_service.py`)
**Endpoints:**
- Upload video for editing
- Apply editing presets
- Generate stock reel from prompts
- Download final output

**Implementation:**
```python
class SubmagicService:
    BASE_URL = "https://api.submagic.co/v1"

    async def edit_reel(self, video_path: str, style: str) -> dict:
        """
        Uploads and edits a video with Submagic.
        """
        # Implement upload and editing logic
        pass

    async def create_stock_reel(self, prompt: str, duration: int) -> dict:
        """
        Generates a stock footage reel.
        """
        pass
```

#### 4.3 Pipeline Execution Agent (`agents/pipeline_execution_agent.py`)
**Workflow Orchestration:**
```python
class PipelineExecutionAgent(BaseReelAgent):
    def __init__(self):
        self.heygen = HeyGenService()
        self.submagic = SubmagicService()
        super().__init__(
            role="Pipeline Executor",
            goal="Execute the chosen reel generation workflow efficiently",
            backstory="Expert in API orchestration and workflow automation"
        )

    async def execute_avatar_workflow(self, avatar_id: str, script: str):
        # 1. Generate avatar video (HeyGen)
        # 2. Add subtitles/effects (Submagic)
        # 3. Download final output
        pass

    async def execute_edited_workflow(self, video_path: str):
        # 1. Upload to Submagic
        # 2. Apply editing
        # 3. Download final output
        pass
```

---

### PHASE 5: Advanced Features (Days 11-12)

#### 5.1 Bulk Processing Mode (`main.py`)
**Features:**
- Process multiple videos in parallel
- Enforce 50/50 avatar/edited split
- Progress tracking
- Error handling and retry logic

**Implementation:**
```python
import asyncio
from typing import List

async def bulk_process(video_paths: List[str], selected_avatar: str):
    """
    Processes multiple videos in bulk mode.
    """
    total = len(video_paths)
    avatar_count = total // 2
    edited_count = total - avatar_count

    # Split videos
    avatar_videos = video_paths[:avatar_count]
    edited_videos = video_paths[avatar_count:]

    # Process in parallel (with concurrency limit)
    semaphore = asyncio.Semaphore(Settings.MAX_PARALLEL_JOBS)

    async def process_with_semaphore(coro):
        async with semaphore:
            return await coro

    # Create tasks
    tasks = []
    for video in avatar_videos:
        tasks.append(process_with_semaphore(
            pipeline_agent.execute_avatar_workflow(selected_avatar, video)
        ))

    for video in edited_videos:
        tasks.append(process_with_semaphore(
            pipeline_agent.execute_edited_workflow(video)
        ))

    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

#### 5.2 Progress Tracking
- Real-time progress bars (using `tqdm`)
- Status updates for each job
- Credit consumption tracking

---

### PHASE 6: Quality Control & Finalization (Days 13-14)

#### 6.1 Quality Check Agent (`agents/quality_check_agent.py`)
**Validation Checks:**
- Video duration (30-90 seconds optimal)
- Subtitle synchronization
- Audio quality
- Resolution standards
- File corruption check

**Implementation:**
```python
class QualityCheckAgent(BaseReelAgent):
    def __init__(self):
        super().__init__(
            role="Quality Assurance Specialist",
            goal="Ensure all generated reels meet quality standards",
            backstory="Perfectionist with expertise in video production quality"
        )

    def validate_reel(self, video_path: str) -> dict:
        """
        Runs quality checks on generated reel.

        Returns:
            Validation results with pass/fail status
        """
        checks = {
            "duration_ok": self._check_duration(video_path),
            "has_subtitles": self._check_subtitles(video_path),
            "resolution_ok": self._check_resolution(video_path),
            "audio_quality": self._check_audio(video_path)
        }

        return {
            "passed": all(checks.values()),
            "details": checks
        }
```

#### 6.2 Output Packaging (`utils/file_handler.py`)
**Output Structure:**
```python
import json
from datetime import datetime
from pathlib import Path

class OutputHandler:
    def __init__(self):
        self.output_dir = Path(Settings.OUTPUT_DIR)

    def create_output_folder(self) -> Path:
        """
        Creates a timestamped folder for reel output.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.output_dir / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_reel_package(self, folder: Path, video_path: str,
                          subtitles: str, metadata: dict):
        """
        Saves all reel components to output folder.
        """
        # Copy final video
        shutil.copy(video_path, folder / "final.mp4")

        # Save subtitles
        (folder / "subtitles.srt").write_text(subtitles)

        # Save metadata
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Generate and save thumbnail
        self._generate_thumbnail(video_path, folder / "thumbnail.jpg")
```

---

## Main Application Flow

### Entry Point (`main.py`)
```python
import asyncio
from utils.media_scanner import scan_video_folder
from core.decision_logic import get_reel_options
from agents.avatar_selector_agent import AvatarSelectorAgent
from agents.pipeline_execution_agent import PipelineExecutionAgent
from agents.quality_check_agent import QualityCheckAgent
from utils.file_handler import OutputHandler

async def main():
    print("=== AI Reel Generation Agent ===\n")

    # 1. Scan for videos
    folder_path = input("Enter video folder path (or press Enter for stock reel): ")

    if folder_path:
        video_count, video_paths = scan_video_folder(folder_path)
        print(f"Found {video_count} videos")
    else:
        video_count, video_paths = 0, []

    # 2. Get reel options
    options = get_reel_options(video_count)
    print(f"\nMode: {options['mode']}")
    print(f"Credits required: {options.get('credits_required', 'TBD')}")

    # 3. Process based on mode
    if options["mode"] == "single":
        if "avatar_reel" in options["options"]:
            avatar_agent = AvatarSelectorAgent()
            selected_avatar = avatar_agent.present_avatars()

        pipeline = PipelineExecutionAgent()
        result = await pipeline.execute_workflow(video_paths, selected_avatar)

    elif options["mode"] == "bulk":
        avatar_agent = AvatarSelectorAgent()
        selected_avatar = avatar_agent.present_avatars()

        results = await bulk_process(video_paths, selected_avatar)

    # 4. Quality check
    qc_agent = QualityCheckAgent()
    validation = qc_agent.validate_reel(result["video_path"])

    if validation["passed"]:
        print("\n✓ Quality check passed!")

        # 5. Save output
        output_handler = OutputHandler()
        folder = output_handler.create_output_folder()
        output_handler.save_reel_package(folder, result["video_path"],
                                         result["subtitles"], result["metadata"])
        print(f"Reel saved to: {folder}")
    else:
        print("\n✗ Quality check failed:", validation["details"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing Strategy

### Unit Tests
- Test media scanner with various folder structures
- Validate decision logic for all video count scenarios
- Test API service methods with mocked responses

### Integration Tests
- End-to-end workflow for each reel type
- Bulk processing with parallel execution
- Error handling and retry mechanisms

### Test Files Structure
```
tests/
├── test_media_scanner.py
├── test_decision_logic.py
├── test_agents.py
├── test_services.py
└── test_integration.py
```

---

## Deployment Checklist

- [ ] Environment variables configured (.env file)
- [ ] API keys validated
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Output directory writable
- [ ] Test with sample videos
- [ ] Error logging configured
- [ ] Rate limiting implemented for APIs
- [ ] Credit tracking system in place

---

## Future Enhancements

1. **Web Interface**: Flask/FastAPI dashboard for easier interaction
2. **Database**: Store reel history, user preferences, avatar choices
3. **Advanced Analytics**: Track performance metrics of generated reels
4. **Custom Avatars**: Allow users to create/upload custom avatars
5. **Multi-Platform Export**: Auto-format for Instagram, TikTok, YouTube Shorts
6. **Scheduling**: Queue reels for automatic posting
7. **A/B Testing**: Generate variants for split testing

---

## Troubleshooting Guide

### Common Issues

**Issue**: API authentication fails
- **Solution**: Verify API keys in `.env`, check key permissions

**Issue**: Video processing timeout
- **Solution**: Increase timeout settings, check internet connection

**Issue**: Quality check fails
- **Solution**: Review validation criteria, check source video quality

**Issue**: Parallel processing errors
- **Solution**: Reduce `MAX_PARALLEL_JOBS`, check system resources

---

## Credits & Cost Management

### Credit Calculation
```python
def calculate_credits(reel_type: str, quantity: int = 1) -> int:
    costs = Settings.CREDIT_COSTS
    return costs.get(reel_type, 0) * quantity
```

### Usage Tracking
- Log every API call with credit cost
- Maintain running balance
- Alert when credits are low
- Generate usage reports

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building a production-ready AI Reel Generation Agent. The modular architecture allows for easy extension and maintenance, while the CrewAI framework enables intelligent, autonomous decision-making throughout the reel generation process.

**Estimated Timeline**: 14 days for full implementation
**Team Size**: 1-2 developers
**Skill Level Required**: Intermediate Python, API integration experience
