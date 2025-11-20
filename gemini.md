# AI Reel Generation Agent - Implementation Plan

## Objective
To implement the AI Reel Generation Agent as described in the project documentation, using Python, CrewAI, and the HeyGen and Submagic APIs.

## Implementation Plan

### PHASE 1 — Core Logic & Project Setup (Day 1–2)
1.  **Initialize Project:**
    *   Create project directory: `reel_generator`.
    *   Set up a Python virtual environment inside `reel_generator`.
    *   Create initial file structure:
        *   `main.py` (entry point)
        *   `requirements.txt`
        *   `config/settings.py` (for API keys, constants)
        *   `utils/media_scanner.py`
        *   `core/decision_logic.py`
        *   `data/avatars.py`

2.  **Configuration Module (`config/settings.py`):**
    *   Define placeholder variables for API keys.
    *   Store credit costs and other static configuration.

3.  **Media Scanner (`utils/media_scanner.py`):**
    *   Implement `scan_video_folder(path)` to count video files in a given directory.

4.  **Decision Logic (`core/decision_logic.py`):**
    *   Implement `get_reel_options(video_count)` to return available reel types and their credit costs based on the business logic.

5.  **Avatar Data (`data/avatars.py`):**
    *   Store the list of available avatars as a Python list of dictionaries.

### PHASE 2 — CrewAI Agent Foundation (Day 3–4)
1.  **Install Dependencies:**
    *   Add `crewai` and any other necessary libraries to `requirements.txt`.
    *   Run `pip install -r requirements.txt`.
2.  **Agent Definitions:**
    *   Create an `agents/` directory.
    *   Define the following agent files with basic class structures inheriting from `crewai.Agent`:
        *   `media_analyzer_agent.py`
        *   `strategy_agent.py`
        *   `script_generation_agent.py`

### PHASE 3 — Avatar Selection (Day 5–6)
1.  **Avatar Selector Agent (`agents/avatar_selector_agent.py`):**
    *   Implement the agent.
    *   Add logic to present avatar choices to the user (e.g., via command-line prompt).
    *   Store the user's selection.

### PHASE 4 — API Integration (Day 7–10)
1.  **API Services:**
    *   Create a `services/` directory.
    *   `heygen_service.py`: Implement functions to interact with the HeyGen API (e.g., `create_avatar_video`).
    *   `submagic_service.py`: Implement functions to interact with the Submagic API (e.g., `edit_reel`, `create_stock_reel`).
2.  **Pipeline Execution Agent (`agents/pipeline_execution_agent.py`):**
    *   Implement agent to call the appropriate service (`HeyGen` or `Submagic`) based on the workflow decision.

### PHASE 5 — Advanced Features (Day 11–12)
1.  **Bulk Mode Logic (`main.py`):**
    *   Implement the main loop to handle multiple video processing sessions.
    *   Add logic for the 50/50 split between Avatar and Edited/Stock reels in bulk mode.
2.  **Parallel Processing:**
    *   Integrate `asyncio` or `multiprocessing` to handle parallel API calls for bulk jobs.

### PHASE 6 — Quality Control & Finalization (Day 13–14)
1.  **QC Agent (`agents/quality_check_agent.py`):**
    *   Implement basic quality checks (e.g., video duration, subtitle presence).
2.  **Output Packaging:**
    *   Implement file I/O logic to create a unique folder for each generated reel.
    *   Save the final `final.mp4`, `subtitles.srt`, `thumbnail.jpg`, and `metadata.json` files to the appropriate folder.
