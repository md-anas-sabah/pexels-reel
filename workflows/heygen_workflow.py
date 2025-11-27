"""
HeyGen Avatar + B-Roll Workflow
Creates YouTube-style explainer videos with AI avatar over B-roll footage
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict
import tempfile

from config import Config
from services.pexels_service import PexelsService
from services.heygen_service import HeyGenService
from services.submagic_service import SubmagicService
from utils.file_uploader import FileUploader
from utils.video_processor import VideoProcessor


logger = logging.getLogger(__name__)


class HeyGenWorkflow:
    """
    Workflow for creating avatar videos with B-roll background

    Steps:
    1. User provides script and selects avatar
    2. Extract keywords from script → Search Pexels for B-roll
    3. Download, trim, concatenate B-roll videos
    4. Upload B-roll to Supabase (get public URL)
    5. Generate HeyGen avatar video with B-roll background
    6. Submit to Submagic for subtitles + effects
    7. Download final video
    """

    def __init__(self):
        """Initialize HeyGen workflow with required services"""
        self.pexels = PexelsService()
        self.heygen = HeyGenService()
        self.submagic = SubmagicService()
        self.uploader = FileUploader()
        self.processor = VideoProcessor()

        self.temp_dir = Path(tempfile.mkdtemp(prefix="heygen_workflow_"))
        logger.info(f"HeyGen workflow temp directory: {self.temp_dir}")

    def display_avatar_menu(self) -> Dict:
        """
        Display avatar selection menu

        Returns:
            Selected avatar dict with id, name, and gender
        """
        print("\n" + "=" * 80)
        print("🎭 SELECT YOUR AVATAR")
        print("=" * 80)

        for i, avatar in enumerate(Config.HEYGEN_AVATARS, 1):
            gender_icon = "👨" if avatar["gender"] == "male" else "👩"
            print(f"  {i}. {gender_icon} {avatar['name']}")

        print("=" * 80)

        while True:
            try:
                choice = input("\n👉 Select avatar (1-{}): ".format(len(Config.HEYGEN_AVATARS)))
                choice_num = int(choice)

                if 1 <= choice_num <= len(Config.HEYGEN_AVATARS):
                    selected = Config.HEYGEN_AVATARS[choice_num - 1]
                    print(f"\n✅ Selected: {selected['name']}")
                    return selected
                else:
                    print(f"❌ Please enter a number between 1 and {len(Config.HEYGEN_AVATARS)}")

            except ValueError:
                print("❌ Please enter a valid number")

    def extract_keywords_from_script(self, script: str, max_keywords: int = 5) -> List[str]:
        """
        Extract relevant keywords from script for B-roll search

        Args:
            script: Avatar script text
            max_keywords: Maximum number of keywords to extract (default: 5)

        Returns:
            List of keywords
        """
        print(f"\n🔍 Extracting keywords from script...")

        # Simple keyword extraction (can be enhanced with NLP)
        # Remove common words and extract nouns/important terms

        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Clean and tokenize
        words = re.findall(r'\b[a-z]+\b', script.lower())

        # Filter out stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Count frequency
        from collections import Counter
        word_counts = Counter(keywords)

        # Get most common keywords
        top_keywords = [word for word, _ in word_counts.most_common(max_keywords)]

        print(f"   Keywords extracted: {', '.join(top_keywords)}")

        return top_keywords

    def prepare_broll(
        self,
        keywords: List[str],
        duration: int = 24,
        clips_per_keyword: int = 2,
        target_clips: int = 6,
        clip_duration: float = 4.0
    ) -> str:
        """
        Prepare B-roll video from Pexels

        Args:
            keywords: Search keywords
            duration: Target total duration in seconds (default: 24)
            clips_per_keyword: Number of clips per keyword (default: 2)
            target_clips: Exact number of clips needed (default: 6)
            clip_duration: Duration per clip in seconds (default: 4.0)

        Returns:
            Path to concatenated B-roll video
        """
        print(f"\n🎬 Preparing B-roll video...")
        print(f"   Keywords: {', '.join(keywords)}")
        print(f"   Target: {target_clips} clips × {clip_duration}s = {duration}s total")

        # Search for videos using multiple keywords
        all_videos = self.pexels.search_multiple_keywords(
            keywords=keywords,
            per_keyword=clips_per_keyword,
            orientation="portrait"
        )

        if not all_videos:
            raise Exception("No B-roll videos found for the given keywords")

        # Download EXACTLY target_clips videos
        print(f"\n📥 Downloading {target_clips} B-roll clips...")
        downloaded_paths = []

        for i, video in enumerate(all_videos[:target_clips], 1):
            output_path = self.temp_dir / f"broll_clip_{i}.mp4"

            try:
                self.pexels.download_video(video['download_url'], str(output_path))
                downloaded_paths.append(str(output_path))
            except Exception as e:
                logger.warning(f"Failed to download clip {i}: {str(e)}")
                continue

        if not downloaded_paths:
            raise Exception("Failed to download any B-roll videos")

        # Use fixed clip_duration (4 seconds per clip)
        print(f"\n📊 Clip Duration: {len(downloaded_paths)} clips × {clip_duration}s = {len(downloaded_paths) * clip_duration}s total")

        # Trim videos to EXACTLY clip_duration
        print(f"\n✂️  Trimming B-roll clips to {clip_duration}s each...")
        trimmed_paths = []

        for i, video_path in enumerate(downloaded_paths, 1):
            trimmed_path = self.temp_dir / f"broll_trimmed_{i}.mp4"

            try:
                self.processor.trim_video(
                    input_path=video_path,
                    output_path=str(trimmed_path),
                    duration=clip_duration
                )
                trimmed_paths.append(str(trimmed_path))
            except Exception as e:
                logger.warning(f"Failed to trim clip {i}: {str(e)}")
                continue

        if not trimmed_paths:
            raise Exception("Failed to trim any B-roll clips")

        # Concatenate videos
        broll_path = self.temp_dir / "broll_concatenated.mp4"
        self.processor.concatenate_videos(trimmed_paths, str(broll_path))

        print(f"\n✅ B-roll prepared: {broll_path}")

        return str(broll_path)

    def execute(
        self,
        script: str,
        avatar_id: str,
        voice_id: str,
        output_path: str,
        title: Optional[str] = None,
        emotion: str = "Excited",
        broll_duration: int = 30,
        ai_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Execute complete HeyGen workflow

        Args:
            script: Text for avatar to speak
            avatar_id: HeyGen avatar ID
            voice_id: HeyGen voice ID
            output_path: Path to save final video
            title: Optional video title
            emotion: Voice emotion (default: "Excited")
            broll_duration: B-roll duration in seconds (default: 30)
            ai_keywords: Pre-generated keywords from AI Agent (optional)

        Returns:
            Path to final video
        """
        print("\n" + "=" * 80)
        print("🎬 HEYGEN AVATAR + B-ROLL WORKFLOW")
        print("=" * 80)

        try:
            # Step 1: Use AI keywords if provided, otherwise extract from script
            # Step 2: Generate HeyGen avatar video (FULL SCREEN with avatar's original background)
            print(f"\n🎭 Generating HeyGen avatar video (full screen with original background)...")
            heygen_result = self.heygen.generate_video(
                script=script,
                avatar_id=avatar_id,
                voice_id=voice_id,
                background_type=None,  # Use avatar's original background
                background_value=None,
                title=title or "AI Avatar Reel",
                emotion=emotion
            )

            video_id = heygen_result["video_id"]

            # Step 3: Wait for HeyGen completion
            heygen_video_url = self.heygen.wait_for_completion(
                video_id=video_id,
                timeout=Config.HEYGEN_TIMEOUT
            )

            # Step 4: Download HeyGen avatar video
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            heygen_local_path = self.temp_dir / f"heygen_avatar_{timestamp}.mp4"
            self.heygen.download_video(heygen_video_url, str(heygen_local_path))

            # Save HeyGen video to output folder (before Submagic)
            import shutil
            output_dir = Path(Config.OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True)

            heygen_output = output_dir / f"heygen_raw_{timestamp}.mp4"
            shutil.copy2(str(heygen_local_path), str(heygen_output))
            print(f"\n📁 HeyGen video saved: {heygen_output}")

            # Step 5: Add padding to prevent Submagic auto-trim
            # Submagic automatically trims from end, so we add 4s padding to be safe
            print(f"\n🔄 Adding padding to prevent Submagic from trimming content...")
            padded_path = self.temp_dir / f"heygen_padded_{timestamp}.mp4"
            self.processor.add_padding(
                input_path=str(heygen_local_path),
                output_path=str(padded_path),
                padding_duration=4.0  # Add 4 seconds of frozen last frame
            )

            # Step 6: Upload padded video to Supabase for Submagic
            print(f"\n☁️  Uploading padded video to Supabase for Submagic...")
            heygen_supabase_url = self.uploader.upload(str(padded_path))

            # Step 7: Submit to Submagic for B-rolls + subtitles + effects
            print(f"\n✨ Submitting to Submagic for Magic B-rolls and subtitles...")
            print(f"   Video URL: {heygen_supabase_url[:60]}...")
            print("   Settings: Karl template, NO Zooms, Magic B-rolls (20%)")

            try:
                # Submagic will add B-rolls automatically using Magic B-rolls feature
                final_path = self.submagic.process_video(
                    video_url=heygen_supabase_url,
                    output_path=str(output_path),
                    title="AI Generated Avatar Reel",
                    language="en",  # Default to English, can be made dynamic
                    template_name="Karl",  # HARDCODED: Karl template (DO NOT CHANGE)
                    magic_zooms=False,  # No zooms
                    magic_brolls=True,  # Magic B-rolls from Submagic's library
                    magic_brolls_percentage=60,  # 20% B-roll coverage
                    position=70  # Removed from API payload (not supported)
                )

                print(f"\n   ✅ Submagic processing complete!")
                print(f"   📥 Downloaded: {Path(final_path).name}")

                print(f"\n" + "=" * 80)
                print(f"✅ WORKFLOW COMPLETE!")
                print(f"   HeyGen raw: {heygen_output}")
                print(f"   Final (B-rolls + subtitles): {final_path}")
                print("=" * 80)

                return final_path

            except Exception as e:
                logger.warning(f"Submagic failed: {e}")
                print(f"\n⚠️  Submagic processing failed: {e}")
                print(f"   Using HeyGen video without edits: {heygen_output}")

                print(f"\n" + "=" * 80)
                print(f"✅ WORKFLOW COMPLETE (Partial)")
                print(f"   HeyGen raw: {heygen_output}")
                print(f"   ⚠️  Submagic failed - use raw HeyGen video")
                print("=" * 80)

                return str(heygen_output)

        except Exception as e:
            logger.error(f"HeyGen workflow failed: {str(e)}")
            raise

        finally:
            # Cleanup temporary files
            self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

        # Also cleanup video processor temp files
        self.processor.cleanup()


# Example usage
if __name__ == "__main__":
    workflow = HeyGenWorkflow()

    # Display avatar menu
    avatar = workflow.display_avatar_menu()

    # Example script
    script = """
    Welcome to our business platform! Today I want to share how our innovative solutions
    can transform your workflow and boost productivity. Let's explore the key features
    that make our platform stand out in the market.
    """

    # Example voice ID (replace with actual HeyGen voice ID)
    voice_id = "your_heygen_voice_id"

    # Execute workflow
    output = workflow.execute(
        script=script,
        avatar_id=avatar["id"],
        voice_id=voice_id,
        output_path="output/heygen_avatar_reel.mp4",
        title="Business Platform Overview",
        emotion="Excited"
    )

    print(f"Final video: {output}")
