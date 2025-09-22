#!/usr/bin/env python3
"""
AI Video Reel Converter using CrewAI and Pexels API
Converts videos to 9:16 aspect ratio (720x1280) with intelligent scaling and padding
"""

import os
import requests
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tempfile
import json
from urllib.parse import urlparse
import subprocess
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import fal_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Video information data class"""
    id: int
    width: int
    height: int
    duration: int
    url: str
    download_url: str
    photographer: str
    aspect_ratio: float



class PexelsVideoSearchTool(BaseTool):
    """Tool for searching videos on Pexels"""
    name: str = "pexels_video_search"
    description: str = "Search for videos on Pexels using a query keyword"
    
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._base_url = "https://api.pexels.com/videos"
    
    def _run(self, query: str, per_page: int = 5) -> str:
        """Search for videos on Pexels
        
        Args:
            query: Search query for videos
            per_page: Number of videos to fetch (default: 5)
        """
        try:
            headers = {"Authorization": self._api_key}
            params = {
                "query": query,
                "per_page": per_page,
                "min_width": 720,    # Minimum width for 720p
                "min_height": 1280,  # Minimum height for vertical content
                "orientation": "portrait"  # Prefer portrait/vertical videos
            }
            
            response = requests.get(f"{self._base_url}/search", headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = []
            
            for video in data.get("videos", []):
                # Find the best quality video file
                video_files = video.get("video_files", [])
                best_video = None
                
                for vf in video_files:
                    if vf.get("file_type") == "video/mp4" and vf.get("quality") in ["hd", "sd"]:
                        if not best_video or (vf.get("width", 0) > best_video.get("width", 0)):
                            best_video = vf
                
                if best_video:
                    video_info = VideoInfo(
                        id=video["id"],
                        width=best_video["width"],
                        height=best_video["height"],
                        duration=video["duration"],
                        url=video["url"],
                        download_url=best_video["link"],
                        photographer=video["user"]["name"],
                        aspect_ratio=best_video["width"] / best_video["height"]
                    )
                    videos.append(video_info)
            
            return json.dumps([{
                "id": v.id,
                "width": v.width,
                "height": v.height,
                "duration": v.duration,
                "url": v.url,
                "download_url": v.download_url,
                "photographer": v.photographer,
                "aspect_ratio": v.aspect_ratio
            } for v in videos], indent=2)
            
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return f"Error searching videos: {str(e)}"

# ObjectDetectionTool removed - using Pexels API filtering instead

class VideoProcessingTool(BaseTool):
    """Tool for processing videos with scaling to 720x1280 (9:16 aspect ratio)"""
    name: str = "video_processing"
    description: str = "Process video by scaling to 720x1280 (9:16 aspect ratio) without cropping"
    
    def _run(self, video_path: str, output_path: str) -> str:
        """Process video with FFmpeg scaling only
        
        Args:
            video_path: Path to the video file
            output_path: Output path for processed video
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # FFmpeg command for high-quality scaling WITHOUT audio preservation
            # Remove original audio so only TTS audio will be in final output
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:-1:-1:color=black",
                "-an",  # Remove audio stream
                "-c:v", "libx264",  # Video codec
                "-preset", "slower",  # Better quality preset
                "-crf", "18",  # High quality setting (lower = better)
                "-pix_fmt", "yuv420p",  # Compatibility format
                "-profile:v", "high", "-level", "4.0",  # High profile for better quality
                "-movflags", "+faststart",  # Optimize for streaming
                "-y",  # Overwrite output file
                output_path
            ]
            
            logger.info(f"Processing video with command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.dumps({
                    "success": True,
                    "output_path": output_path,
                    "processing_method": "scale_with_padding",
                    "final_resolution": "720x1280"
                })
            else:
                error_msg = result.stderr
                logger.error(f"FFmpeg error: {error_msg}")
                return f"Error processing video: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return f"Error in video processing: {str(e)}"

class FalMusicGenerationTool(BaseTool):
    """Tool for generating music using Sonauto V2.2 via Fal AI"""
    name: str = "fal_music_generation"
    description: str = "Generate background music for videos using Sonauto V2.2"
    
    def __init__(self, fal_key: str):
        super().__init__()
        self._fal_key = fal_key
        # Configure fal_client with the key
        os.environ['FAL_KEY'] = fal_key
    
    def _run(self, prompt_text: str) -> str:
        """Generate music using Sonauto V2.2"""
        try:
            logger.info(f"Generating music with prompt: '{prompt_text}'")
            
            # Call Sonauto V2.2 via fal_client
            result = fal_client.subscribe(
                "sonauto/v2/text-to-music",
                arguments={
                    "prompt": prompt_text,
                    "prompt_strength": 2.0,  # Within valid range (>= 1.4)
                    "balance_strength": 0.7,
                    "num_songs": 1,
                    "output_format": "mp3",
                    "bpm": "auto"
                }
            )
            
            if result and 'audio' in result:
                audio_url = result['audio'][0]['url'] if isinstance(result['audio'], list) else result['audio']['url']
                return json.dumps({
                    "success": True,
                    "audio_url": audio_url,
                    "duration": result.get('duration', 30),
                    "prompt": prompt_text,
                    "type": "music"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "No audio URL returned from Sonauto",
                    "type": "music"
                })
                
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "type": "music"
            })

class FalTTSGenerationTool(BaseTool):
    """Tool for generating Text-to-Speech using Orpheus TTS via Fal AI"""
    name: str = "fal_tts_generation"
    description: str = "Generate narrative voice-over using Orpheus TTS"
    
    def __init__(self, fal_key: str):
        super().__init__()
        self._fal_key = fal_key
        # Configure fal_client with the key
        os.environ['FAL_KEY'] = fal_key
    
    def _run(self, text_to_speak: str) -> str:
        """Generate speech using Orpheus TTS"""
        try:
            # Validate input
            if not text_to_speak or not isinstance(text_to_speak, str):
                return json.dumps({
                    "success": False,
                    "error": "Invalid text input for TTS",
                    "type": "tts"
                })
                
            logger.info(f"Generating TTS for text: '{text_to_speak[:50]}...'")
            
            # Direct Fal API call using the correct format from documentation
            result = fal_client.subscribe(
                "fal-ai/orpheus-tts",
                arguments={
                    "text": text_to_speak,
                    "voice": "tara",
                    "temperature": 0.7,
                    "repetition_penalty": 1.2
                }
            )
            
            logger.info(f"TTS API response: {result}")
            
            if result and 'audio' in result:
                audio_url = result['audio']['url']
                logger.info(f"TTS generated successfully: {audio_url}")
                return json.dumps({
                    "success": True,
                    "audio_url": audio_url,
                    "duration": len(text_to_speak) * 0.1,  # Estimate duration
                    "text": text_to_speak,
                    "voice": "tara",
                    "type": "tts"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "No audio URL returned from Orpheus TTS",
                    "result": result,
                    "type": "tts"
                })
            
                
        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "type": "tts"
            })

class FalSubtitleGenerationTool(BaseTool):
    """Tool for generating subtitles from audio using Fal AI Whisper"""
    name: str = "fal_subtitle_generation"
    description: str = "Generate word-level subtitles from audio using Fal AI Whisper"
    
    def __init__(self, fal_key: str):
        super().__init__()
        self._fal_key = fal_key
        # Configure fal_client with the key
        os.environ['FAL_KEY'] = fal_key
    
    def _run(self, audio_url: str, temp_dir: str) -> str:
        """Generate subtitles from audio using Fal AI Whisper
        
        Args:
            audio_url: URL to the audio file
            temp_dir: Directory to save the SRT file
            
        Returns:
            JSON string with subtitle generation results
        """
        try:
            logger.info(f"Generating subtitles for audio: {audio_url}")
            
            # Call Fal AI Whisper API
            result = fal_client.subscribe(
                "fal-ai/whisper",
                arguments={
                    "audio_url": audio_url,
                    "task": "transcribe",
                    "chunk_level": "word",  # Word-level timestamps
                    "version": "3"
                }
            )
            
            logger.info(f"Whisper transcription completed")
            
            if result and 'chunks' in result:
                # Generate SRT file from chunks
                srt_filename = f"subtitles_{os.getpid()}.srt"
                srt_path = os.path.join(temp_dir, srt_filename)
                
                # Create SRT content
                srt_content = self._create_srt_from_chunks(result['chunks'])
                
                # Write SRT file
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                
                logger.info(f"✅ Subtitles generated: {srt_path}")
                
                return json.dumps({
                    "success": True,
                    "srt_path": srt_path,
                    "text": result.get('text', ''),
                    "word_count": len(result['chunks']) if result['chunks'] else 0,
                    "type": "subtitles"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "No chunks returned from Whisper",
                    "type": "subtitles"
                })
                
        except Exception as e:
            logger.error(f"Error generating subtitles: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "type": "subtitles"
            })
    
    def _create_srt_from_chunks(self, chunks: List[Dict]) -> str:
        """Create SRT subtitle content from Whisper chunks
        
        Args:
            chunks: List of word chunks with timestamps
            
        Returns:
            SRT formatted subtitle content
        """
        srt_content = ""
        subtitle_index = 1
        
        # Group words into subtitle segments (2-4 words per subtitle for readability)
        words_per_subtitle = 3
        
        for i in range(0, len(chunks), words_per_subtitle):
            # Get word group
            word_group = chunks[i:i + words_per_subtitle]
            
            if not word_group:
                continue
            
            # Get timing from first and last word in group
            start_time = word_group[0]['timestamp'][0] if word_group[0].get('timestamp') else 0
            end_time = word_group[-1]['timestamp'][1] if word_group[-1].get('timestamp') else start_time + 1
            
            # Combine text from all words in group
            subtitle_text = ' '.join([word.get('text', '').strip() for word in word_group]).strip()
            
            if subtitle_text:
                # Format timestamps for SRT (HH:MM:SS,mmm)
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                # Add SRT entry
                srt_content += f"{subtitle_index}\n"
                srt_content += f"{start_srt} --> {end_srt}\n"
                srt_content += f"{subtitle_text}\n\n"
                
                subtitle_index += 1
        
        return srt_content
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

class AudioMixingTool(BaseTool):
    """Tool for mixing generated audio with video using FFmpeg"""
    name: str = "audio_mixing"
    description: str = "Mix background music and/or voice-over with video"
    
    def _get_subtitle_style(self) -> str:
        """Get optimized subtitle styling for social media reels based on user-provided image.
        
        Returns:
            ASS subtitle style string for FFmpeg
        """
        # Style to match the user's screenshot: A clean, bold white font with a thin black outline.
        # Font: Arial Bold (for maximum compatibility and similar look to the image)
        # Outline: A 2px black outline with no background or shadow.
        # Position: Bottom-center with a vertical margin.
        
        font_name = 'Arial'
        
        return (
            f"Fontname={font_name},"
            "Fontsize=22,"
            "Bold=1,"
            "PrimaryColour=&HFFFFFF&,"      # White text
            "OutlineColour=&H000000&,"      # Black outline
            "BorderStyle=1,"                # Use 1 for outline WITHOUT a background box
            "Outline=2,"                    # 2px outline width
            "Shadow=0,"                     # No shadow, for a clean outline
            "Alignment=2,"                  # Bottom-center alignment
            "MarginV=80"                    # Vertical margin from the bottom
        )
    
    def _run(self, video_path: str, audio_url: str, output_path: str, srt_path: str = None) -> str:
        """Mix audio with video using FFmpeg, optionally burning subtitles"""
        try:
            # Download audio file
            temp_audio_path = f"/tmp/temp_audio_{os.getpid()}.mp3"
            
            # Download the audio
            response = requests.get(audio_url)
            with open(temp_audio_path, 'wb') as f:
                f.write(response.content)
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-i", video_path, "-i", temp_audio_path]
            
            # Add subtitle filter if SRT file is provided
            if srt_path and os.path.exists(srt_path):
                # Escape the SRT path for FFmpeg (handle special characters)
                srt_path_escaped = srt_path.replace('\\', '/').replace(':', '\\:')
                
                # Professional subtitle filter with optimized styling
                subtitle_style = self._get_subtitle_style()
                subtitle_filter = f"subtitles={srt_path_escaped}:force_style='{subtitle_style}'"
                
                cmd.extend(["-vf", subtitle_filter])
                cmd.extend(["-c:v", "libx264", "-preset", "slower", "-crf", "18"])
                logger.info(f"🎬 Adding clean subtitles: {srt_path}")
                logger.info(f"📝 Subtitle styling: Montserrat Bold, 26px, white text with black stroke (no background)")
            else:
                # No subtitles, copy video without re-encoding
                cmd.extend(["-c:v", "copy"])
            
            # Audio settings
            cmd.extend([
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",  # Match the duration of the shorter stream
                "-y", output_path
            ])
            
            logger.info(f"Mixing audio with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Clean up temp audio file
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
                
                return json.dumps({
                    "success": True,
                    "output_path": output_path,
                    "message": "Audio mixed successfully",
                    "subtitles_burned": srt_path is not None
                })
            else:
                error_msg = result.stderr
                logger.error(f"Audio mixing error: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Audio mixing failed: {error_msg}"
                })
                
        except Exception as e:
            logger.error(f"Error in audio mixing: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })

class VideoReelConverter:
    """Main class for converting videos to reel format using CrewAI"""
    
    def __init__(self, pexels_api_key: str, fal_key: Optional[str] = None):
        self.pexels_api_key = pexels_api_key
        self.fal_key = fal_key or os.getenv('FAL_KEY')
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = "output_reels"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tools
        self.pexels_tool = PexelsVideoSearchTool(pexels_api_key)
        self.processing_tool = VideoProcessingTool()
        
        # Initialize audio tools if Fal key is available
        if self.fal_key:
            self.music_tool = FalMusicGenerationTool(self.fal_key)
            self.tts_tool = FalTTSGenerationTool(self.fal_key)
            self.subtitle_tool = FalSubtitleGenerationTool(self.fal_key)
            self.audio_mixing_tool = AudioMixingTool()
        else:
            self.music_tool = None
            self.tts_tool = None
            self.subtitle_tool = None
            self.audio_mixing_tool = None
            logger.warning("FAL_KEY not provided. Audio generation will be disabled.")
        
        # Initialize agents
        self._setup_agents()
        
    def _setup_agents(self):
        """Setup CrewAI agents"""
        
        # Video Search Agent
        self.search_agent = Agent(
            role="Video Content Researcher",
            goal="Find high-quality videos from Pexels that match user requirements",
            backstory="""You are an expert at finding the perfect video content. You understand 
            what makes a good video for social media reels and can identify videos that will 
            work well when converted to vertical format.""",
            tools=[self.pexels_tool],
            verbose=True
        )
        
# Object Detection Agent removed - using Pexels API filtering instead
        
        # Video Processing Agent
        self.processing_agent = Agent(
            role="Video Production Expert",
            goal="Convert videos to perfect reel format with optimal scaling and quality",
            backstory="""You are a video production specialist who creates high-quality vertical 
            videos for social media. You ensure videos are properly scaled to 720x1280 resolution 
            while maintaining aspect ratio and optimizing quality.""",
            tools=[self.processing_tool],
            verbose=True
        )
        
        # Audio Production Agent (only if audio tools are available)
        if self.music_tool and self.tts_tool:
            audio_tools = [self.music_tool, self.tts_tool, self.audio_mixing_tool]
            self.audio_agent = Agent(
                role="Audio Production Specialist",
                goal="Create engaging background music and voice-overs for social media reels",
                backstory="""You are an audio production expert who specializes in creating 
                perfect soundtracks for social media content. You understand how to match music 
                styles with video content and create compelling voice-overs that enhance viewer 
                engagement. You know how to balance audio levels for optimal social media playback.""",
                tools=audio_tools,
                verbose=True
            )
        else:
            self.audio_agent = None
    
    def download_video(self, video_url: str, filename: str) -> str:
        """Download video from URL"""
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            file_path = os.path.join(self.temp_dir, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video downloaded: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    
    def _setup_search_crew(self, query: str, per_page: int = 5):
        """Setup search crew for finding videos"""
        search_task = Task(
            description=f"""Search for {per_page} high-quality videos on Pexels using the query: "{query}". 
            Focus on videos that would work well for vertical social media content. Return detailed 
            information about each video including download URLs and dimensions.""",
            agent=self.search_agent,
            expected_output="JSON list of video information including IDs, dimensions, download URLs, and metadata"
        )
        
        return Crew(
            agents=[self.search_agent],
            tasks=[search_task],
            process=Process.sequential,
            verbose=True
        )
    
    def _process_single_video(self, video_data: Dict, query: str) -> Optional[Dict]:
        """Process a single video"""
        try:
            # Download video
            video_filename = f"video_{video_data['id']}.mp4"
            video_path = self.download_video(video_data['download_url'], video_filename)
            
            # Object detection
            detection_task = Task(
                description=f"""Analyze the video at {video_path} to detect faces, people, and important 
                objects. Determine the optimal cropping region for converting this video to 9:16 aspect 
                ratio while preserving the most important visual elements.""",
                agent=self.detection_agent,
                expected_output="JSON object with ROI coordinates and detection information"
            )
            
            detection_crew = Crew(
                agents=[self.detection_agent],
                tasks=[detection_task],
                process=Process.sequential,
                verbose=True
            )
            
            detection_result = detection_crew.kickoff()
            
            # Video processing
            output_filename = f"reel_{video_data['id']}_{query.replace(' ', '_')}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            processing_task = Task(
                description=f"""Process the video at {video_path} using the ROI data: {detection_result}. 
                Crop the video to the optimal region and resize it to exactly 720x1280 pixels (9:16 aspect ratio). 
                Preserve audio quality and optimize for social media playback. Save to {output_path}.""",
                agent=self.processing_agent,
                expected_output="JSON object with processing results and output file information"
            )
            
            processing_crew = Crew(
                agents=[self.processing_agent],
                tasks=[processing_task],
                process=Process.sequential,
                verbose=True
            )
            
            processing_result = processing_crew.kickoff()
            processing_data = json.loads(str(processing_result))
            
            if processing_data.get("success"):
                return {
                    "original_video": video_data,
                    "detection_result": json.loads(str(detection_result)),
                    "processing_result": processing_data,
                    "output_file": output_path,
                    "photographer_credit": f"Video by {video_data['photographer']} on Pexels"
                }
            
        except Exception as e:
            logger.error(f"Error processing video {video_data.get('id', 'unknown')}: {e}")
        
        return None

    def _process_single_video_with_audio(self, video_data: Dict, query: str, 
                                       audio_options: Dict = None) -> Optional[Dict]:
        """Process a single video with optional audio generation"""
        try:
            # Download video
            video_filename = f"video_{video_data['id']}.mp4"
            video_path = self.download_video(video_data['download_url'], video_filename)
            
            # Video processing (save to unaudio_video folder first) - no cropping, just scaling
            unaudio_filename = f"reel_{video_data['id']}_{query.replace(' ', '_')}_unaudio.mp4"
            unaudio_dir = os.path.join(self.output_dir, "unaudio_video")
            unaudio_output_path = os.path.join(unaudio_dir, unaudio_filename)
            
            # Ensure unaudio_video directory exists
            os.makedirs(unaudio_dir, exist_ok=True)
            
            processing_task = Task(
                description=f"""Process the video at {video_path} by scaling it to exactly 720x1280 pixels (9:16 aspect ratio). 
                Maintain aspect ratio with padding if needed. Preserve audio quality and optimize for social media playback. 
                Save to {unaudio_output_path}.""",
                agent=self.processing_agent,
                expected_output="JSON object with processing results and output file information"
            )
            
            processing_crew = Crew(
                agents=[self.processing_agent],
                tasks=[processing_task],
                process=Process.sequential,
                verbose=True
            )
            
            processing_result = processing_crew.kickoff()
            processing_data = json.loads(str(processing_result))
            
            if not processing_data.get("success"):
                return None
            
            # Log that the unaudio video has been saved
            logger.info(f"✅ SCALED VIDEO SAVED: {unaudio_output_path}")
            logger.info(f"🎬 You can now check the video quality at: {unaudio_output_path}")
            logger.info(f"⏹️  To stop before audio generation, press Ctrl+C now!")
            
            # Audio generation and mixing (if requested)
            final_output_filename = f"reel_{video_data['id']}_{query.replace(' ', '_')}.mp4"
            final_output_path = os.path.join(self.output_dir, final_output_filename)
            
            audio_results = []
            
            if audio_options and self.audio_agent:
                # Generate music if requested - using direct API call to bypass CrewAI issues
                if audio_options.get('music', False):
                    music_style = audio_options.get('music_style', 'upbeat energetic')
                    prompt_text = f"{music_style} instrumental music for {query} video reel suitable for social media"
                    
                    # Direct music generation bypassing CrewAI to avoid parameter issues
                    logger.info(f"Generating music directly: {prompt_text}")
                    try:
                        music_tool = FalMusicGenerationTool(self.fal_key)
                        music_result = music_tool._run(prompt_text)
                        music_data = json.loads(music_result)
                        if music_data.get("success"):
                            audio_results.append(music_data)
                        logger.info(f"Direct music result: {music_result}")
                    except Exception as e:
                        logger.error(f"Direct music generation failed: {e}")
                
                # Generate voice-over if requested
                if audio_options.get('voice', False) and audio_options.get('voice_text'):
                    voice_text = audio_options['voice_text']
                    voice_style = audio_options.get('voice_style', 'neutral')
                    
                    
                    # Direct TTS generation bypassing CrewAI to avoid parameter issues
                    logger.info(f"Generating TTS directly: {voice_text}")
                    try:
                        tts_tool = FalTTSGenerationTool(self.fal_key)
                        voice_result = tts_tool._run(voice_text)
                        voice_data = json.loads(voice_result)
                        if voice_data.get("success"):
                            # Generate subtitles for TTS audio (MANDATORY when TTS is used)
                            logger.info("🎬 Generating subtitles for TTS audio...")
                            try:
                                subtitle_tool = FalSubtitleGenerationTool(self.fal_key)
                                subtitle_result = subtitle_tool._run(
                                    audio_url=voice_data['audio_url'],
                                    temp_dir=self.temp_dir
                                )
                                subtitle_data = json.loads(subtitle_result)
                                if subtitle_data.get("success"):
                                    voice_data['srt_path'] = subtitle_data['srt_path']
                                    logger.info(f"✅ Subtitles generated: {subtitle_data['srt_path']}")
                                else:
                                    logger.error(f"❌ Subtitle generation failed: {subtitle_data.get('error')}")
                            except Exception as subtitle_error:
                                logger.error(f"❌ Subtitle generation error: {subtitle_error}")
                            
                            audio_results.append(voice_data)
                        logger.info(f"Direct TTS result: {voice_result}")
                    except Exception as e:
                        logger.error(f"Direct TTS generation failed: {e}")
                
                # Mix audio with video if we have any audio
                if audio_results:
                    # Check if unaudio video file exists
                    if not os.path.exists(unaudio_output_path):
                        logger.error(f"❌ Unaudio video file not found: {unaudio_output_path}")
                        logger.error("Cannot proceed with audio mixing. Video processing may have failed.")
                        return None
                    
                    current_video_path = unaudio_output_path
                    
                    for i, audio_data in enumerate(audio_results):
                        if i == len(audio_results) - 1:
                            # Last audio mixing - use final output path
                            output_path = final_output_path
                        else:
                            # Intermediate mixing
                            output_path = os.path.join(self.temp_dir, f"mixed_{i}_{video_data['id']}.mp4")
                        
                        # Direct audio mixing bypassing CrewAI to avoid issues
                        logger.info(f"Mixing audio directly: {audio_data['audio_url']} with {current_video_path}")
                        
                        # Double check current video path exists before mixing
                        if not os.path.exists(current_video_path):
                            logger.error(f"❌ Video file missing during mixing: {current_video_path}")
                            return None
                        
                        try:
                            audio_mixing_tool = AudioMixingTool()
                            # Get SRT path if this is TTS audio with subtitles
                            srt_path = audio_data.get('srt_path') if audio_data.get('type') == 'tts' else None
                            
                            mixing_result = audio_mixing_tool._run(
                                video_path=current_video_path,
                                audio_url=audio_data['audio_url'],
                                output_path=output_path,
                                srt_path=srt_path
                            )
                            mixing_data = json.loads(mixing_result)
                            
                            if mixing_data.get("success"):
                                # Verify the output file actually exists
                                if os.path.exists(output_path):
                                    current_video_path = output_path
                                    logger.info(f"✅ Audio mixing successful: {output_path}")
                                else:
                                    logger.error(f"❌ Audio mixing claimed success but output file missing: {output_path}")
                                    return None
                            else:
                                logger.error(f"❌ Audio mixing failed: {mixing_data}")
                                return None
                        except Exception as e:
                            logger.error(f"❌ Direct audio mixing failed: {e}")
                            return None
                else:
                    # No audio requested - just copy unaudio video to final output
                    if os.path.exists(unaudio_output_path):
                        import shutil
                        shutil.copy2(unaudio_output_path, final_output_path)
                    else:
                        logger.error(f"❌ Cannot copy unaudio video - file not found: {unaudio_output_path}")
                        return None
            else:
                # No audio generation available - just copy unaudio video to final output
                if os.path.exists(unaudio_output_path):
                    import shutil
                    shutil.copy2(unaudio_output_path, final_output_path)
                else:
                    logger.error(f"❌ Cannot copy unaudio video - file not found: {unaudio_output_path}")
                    return None
            
            return {
                "original_video": video_data,
                "processing_result": processing_data,
                "audio_results": audio_results,
                "unaudio_file": unaudio_output_path,
                "output_file": final_output_path,
                "photographer_credit": f"Video by {video_data['photographer']} on Pexels"
            }
            
        except Exception as e:
            logger.error(f"Error processing video with audio {video_data.get('id', 'unknown')}: {e}")
            return None

    def trim_segment(self, video_path: str, start_time: float, duration: float, output_path: str) -> bool:
        """Trim a segment from a video using FFmpeg
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            duration: Duration of segment in seconds
            output_path: Path for trimmed segment
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),  # Start time
                "-t", str(duration),     # Duration
                "-c:v", "libx264",       # Video codec
                "-preset", "faster",     # Faster preset for trimming
                "-crf", "18",           # High quality
                "-c:a", "aac",          # Audio codec (keep audio for trimming)
                "-y", output_path
            ]
            
            logger.info(f"Trimming segment: {start_time}s-{start_time+duration}s from {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Segment trimmed: {output_path}")
                return True
            else:
                logger.error(f"❌ Trimming failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error trimming segment: {e}")
            return False

    def concat_clips(self, clip_paths: List[str], output_path: str) -> bool:
        """Concatenate multiple video clips using FFmpeg concat filter
        
        Args:
            clip_paths: List of paths to video clips
            output_path: Path for concatenated output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(clip_paths) < 2:
                logger.error("Need at least 2 clips to concatenate")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build FFmpeg command with concat filter (VIDEO ONLY - no audio)
            inputs = []
            filter_complex = []
            
            # Add input files
            for i, clip_path in enumerate(clip_paths):
                inputs.extend(["-i", clip_path])
                filter_complex.append(f"[{i}:v]")
            
            # Create concat filter for video only (since trimmed clips have no audio)
            concat_filter = f"{''.join(filter_complex)}concat=n={len(clip_paths)}:v=1:a=0[v]"
            
            cmd = [
                "ffmpeg"
            ] + inputs + [
                "-filter_complex", concat_filter,
                "-map", "[v]",  # Only map video, no audio
                "-c:v", "libx264",
                "-preset", "slower",    # High quality for final output
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-level", "4.0",
                "-movflags", "+faststart",
                "-y", output_path
            ]
            
            logger.info(f"Concatenating {len(clip_paths)} video-only clips")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Clips concatenated: {output_path}")
                return True
            else:
                logger.error(f"❌ Concatenation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error concatenating clips: {e}")
            return False

    def fetch_multiple_videos(self, query: str, count: int = 6) -> List[Dict]:
        """Fetch multiple videos for multi-clip reel creation
        
        Args:
            query: Search query
            count: Number of videos to fetch (5-7 recommended)
            
        Returns:
            List of video data dictionaries
        """
        try:
            # Clamp count to reasonable range
            count = max(5, min(count, 7))
            
            search_task = Task(
                description=f"""Search for {count} high-quality videos on Pexels using the query: "{query}". 
                Focus on videos that would work well for vertical social media content and multi-clip reels. 
                Return detailed information about each video including download URLs and dimensions.""",
                agent=self.search_agent,
                expected_output="JSON list of video information including IDs, dimensions, download URLs, and metadata"
            )
            
            search_crew = Crew(
                agents=[self.search_agent],
                tasks=[search_task],
                process=Process.sequential,
                verbose=True
            )
            
            search_result = search_crew.kickoff()
            videos_data = json.loads(str(search_result))
            
            logger.info(f"Fetched {len(videos_data)} videos for multi-clip reel")
            return videos_data
            
        except Exception as e:
            logger.error(f"Error fetching multiple videos: {e}")
            return []

    def convert_to_reel_with_audio(self, query: str, audio_options: Dict = None, per_page: int = 3, mode: str = 'single') -> List[Dict]:
        """Convert videos to reel format with optional audio generation
        
        Args:
            query: Search query for videos
            audio_options: Audio generation options
            per_page: Number of videos to fetch
            mode: 'single' for one video or 'multi' for multi-clip reel
        """
        try:
            results = []
            
            if mode == 'single':
                # SINGLE MODE: Original workflow - process individual videos
                search_task = Task(
                    description=f"""Search for {per_page} high-quality videos on Pexels using the query: "{query}". 
                    Focus on videos that would work well for vertical social media content. Return detailed 
                    information about each video including download URLs and dimensions.""",
                    agent=self.search_agent,
                    expected_output="JSON list of video information including IDs, dimensions, download URLs, and metadata"
                )
                
                search_crew = Crew(
                    agents=[self.search_agent],
                    tasks=[search_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                search_result = search_crew.kickoff()
                videos_data = json.loads(str(search_result))
                
                logger.info(f"[SINGLE MODE] Found {len(videos_data)} videos for processing")
                
                # Process each video individually
                for idx, video_data in enumerate(videos_data):
                    try:
                        logger.info(f"Processing video {idx + 1}/{len(videos_data)}: ID {video_data['id']}")
                        
                        result = self._process_single_video_with_audio(video_data, query, audio_options)
                        if result:
                            results.append(result)
                            logger.info(f"Successfully processed video {video_data['id']}")
                        else:
                            logger.error(f"Failed to process video {video_data['id']}")
                            
                    except Exception as e:
                        logger.error(f"Error processing video {video_data.get('id', 'unknown')}: {e}")
                        continue
                        
            elif mode == 'multi':
                # MULTI MODE: Create one reel from multiple video segments
                logger.info(f"[MULTI MODE] Creating multi-clip reel for query: '{query}'")
                
                # Fetch 5-7 videos for multi-clip
                videos_data = self.fetch_multiple_videos(query, count=6)
                
                if len(videos_data) < 2:
                    logger.error("Need at least 2 videos for multi-clip reel")
                    return []
                
                logger.info(f"[MULTI MODE] Processing {len(videos_data)} videos for multi-clip reel")
                
                # Step 1: Download and process all videos to 720x1280
                processed_clips = []
                clips_dir = os.path.join(self.temp_dir, "clips")
                os.makedirs(clips_dir, exist_ok=True)
                
                for idx, video_data in enumerate(videos_data):
                    try:
                        logger.info(f"Processing clip {idx + 1}/{len(videos_data)}: ID {video_data['id']}")
                        
                        # Download video
                        video_filename = f"video_{video_data['id']}.mp4"
                        video_path = self.download_video(video_data['download_url'], video_filename)
                        
                        # Process to 720x1280
                        processed_filename = f"processed_{video_data['id']}.mp4"
                        processed_path = os.path.join(clips_dir, processed_filename)
                        
                        processing_task = Task(
                            description=f"""Process the video at {video_path} by scaling it to exactly 720x1280 pixels (9:16 aspect ratio). 
                            Maintain aspect ratio with padding if needed. Remove audio since we'll add our own later. 
                            Save to {processed_path}.""",
                            agent=self.processing_agent,
                            expected_output="JSON object with processing results and output file information"
                        )
                        
                        processing_crew = Crew(
                            agents=[self.processing_agent],
                            tasks=[processing_task],
                            process=Process.sequential,
                            verbose=False  # Less verbose for multi-clip
                        )
                        
                        processing_result = processing_crew.kickoff()
                        processing_data = json.loads(str(processing_result))
                        
                        if processing_data.get("success"):
                            processed_clips.append({
                                'path': processed_path,
                                'video_data': video_data,
                                'duration': video_data['duration']
                            })
                            logger.info(f"✅ Processed clip {idx + 1}: {processed_path}")
                        else:
                            logger.error(f"❌ Failed to process clip {idx + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing clip {video_data.get('id', 'unknown')}: {e}")
                        continue
                
                if len(processed_clips) < 2:
                    logger.error("Failed to process enough clips for multi-clip reel")
                    return []
                
                # Step 2: Trim 3-4 second segments from each processed clip
                trimmed_clips = []
                segment_duration = 3.5  # 3.5 seconds per segment
                
                for idx, clip_info in enumerate(processed_clips):
                    try:
                        # Calculate start time (middle of video for best content)
                        video_duration = clip_info['duration']
                        start_time = max(0, (video_duration - segment_duration) / 2)
                        
                        trimmed_filename = f"segment_{idx}_{clip_info['video_data']['id']}.mp4"
                        trimmed_path = os.path.join(clips_dir, trimmed_filename)
                        
                        if self.trim_segment(clip_info['path'], start_time, segment_duration, trimmed_path):
                            trimmed_clips.append(trimmed_path)
                            logger.info(f"✅ Trimmed segment {idx + 1}: {trimmed_path}")
                        else:
                            logger.error(f"❌ Failed to trim segment {idx + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error trimming segment {idx + 1}: {e}")
                        continue
                
                if len(trimmed_clips) < 2:
                    logger.error("Failed to trim enough segments for multi-clip reel")
                    return []
                
                # Step 3: Concatenate trimmed segments
                concat_filename = f"multi_reel_{query.replace(' ', '_')}_concat.mp4"
                concat_dir = os.path.join(self.output_dir, "unaudio_video")
                concat_path = os.path.join(concat_dir, concat_filename)
                
                if self.concat_clips(trimmed_clips, concat_path):
                    logger.info(f"✅ Multi-clip video created: {concat_path}")
                    
                    # Step 4: Apply audio if requested
                    final_output_filename = f"multi_reel_{query.replace(' ', '_')}.mp4"
                    final_output_path = os.path.join(self.output_dir, final_output_filename)
                    
                    audio_results = []
                    
                    if audio_options and self.audio_agent:
                        # Generate music if requested
                        if audio_options.get('music', False):
                            music_style = audio_options.get('music_style', 'upbeat energetic')
                            prompt_text = f"{music_style} instrumental music for {query} multi-clip video reel suitable for social media"
                            
                            try:
                                music_tool = FalMusicGenerationTool(self.fal_key)
                                music_result = music_tool._run(prompt_text)
                                music_data = json.loads(music_result)
                                if music_data.get("success"):
                                    audio_results.append(music_data)
                                logger.info(f"Multi-clip music generated successfully")
                            except Exception as e:
                                logger.error(f"Multi-clip music generation failed: {e}")
                        
                        # Generate voice-over if requested
                        if audio_options.get('voice', False) and audio_options.get('voice_text'):
                            voice_text = audio_options['voice_text']
                            
                            try:
                                tts_tool = FalTTSGenerationTool(self.fal_key)
                                voice_result = tts_tool._run(voice_text)
                                voice_data = json.loads(voice_result)
                                if voice_data.get("success"):
                                    # Generate subtitles for multi-clip TTS audio (MANDATORY when TTS is used)
                                    logger.info("🎬 Generating subtitles for multi-clip TTS audio...")
                                    try:
                                        subtitle_tool = FalSubtitleGenerationTool(self.fal_key)
                                        subtitle_result = subtitle_tool._run(
                                            audio_url=voice_data['audio_url'],
                                            temp_dir=self.temp_dir
                                        )
                                        subtitle_data = json.loads(subtitle_result)
                                        if subtitle_data.get("success"):
                                            voice_data['srt_path'] = subtitle_data['srt_path']
                                            logger.info(f"✅ Multi-clip subtitles generated: {subtitle_data['srt_path']}")
                                        else:
                                            logger.error(f"❌ Multi-clip subtitle generation failed: {subtitle_data.get('error')}")
                                    except Exception as subtitle_error:
                                        logger.error(f"❌ Multi-clip subtitle generation error: {subtitle_error}")
                                    
                                    audio_results.append(voice_data)
                                logger.info(f"Multi-clip TTS generated successfully")
                            except Exception as e:
                                logger.error(f"Multi-clip TTS generation failed: {e}")
                        
                        # Mix audio with concatenated video
                        if audio_results:
                            current_video_path = concat_path
                            
                            for i, audio_data in enumerate(audio_results):
                                if i == len(audio_results) - 1:
                                    output_path = final_output_path
                                else:
                                    output_path = os.path.join(self.temp_dir, f"multi_mixed_{i}.mp4")
                                
                                try:
                                    audio_mixing_tool = AudioMixingTool()
                                    # Get SRT path if this is TTS audio with subtitles
                                    srt_path = audio_data.get('srt_path') if audio_data.get('type') == 'tts' else None
                                    
                                    mixing_result = audio_mixing_tool._run(
                                        video_path=current_video_path,
                                        audio_url=audio_data['audio_url'],
                                        output_path=output_path,
                                        srt_path=srt_path
                                    )
                                    mixing_data = json.loads(mixing_result)
                                    
                                    if mixing_data.get("success"):
                                        current_video_path = output_path
                                        logger.info(f"✅ Multi-clip audio mixing successful")
                                    else:
                                        logger.error(f"❌ Multi-clip audio mixing failed")
                                        return []
                                except Exception as e:
                                    logger.error(f"❌ Multi-clip audio mixing error: {e}")
                                    return []
                        else:
                            # No audio - copy concat video to final output
                            import shutil
                            shutil.copy2(concat_path, final_output_path)
                    else:
                        # No audio generation available - copy concat video to final output
                        import shutil
                        shutil.copy2(concat_path, final_output_path)
                    
                    # Clean up intermediate trimmed clips
                    try:
                        for clip_path in trimmed_clips:
                            if os.path.exists(clip_path):
                                os.remove(clip_path)
                        logger.info("✅ Cleaned up intermediate trimmed clips")
                    except Exception as e:
                        logger.warning(f"Warning: Could not clean up some intermediate files: {e}")
                    
                    # Create result for multi-clip reel
                    multi_result = {
                        "mode": "multi",
                        "videos_used": [clip['video_data'] for clip in processed_clips],
                        "segment_count": len(trimmed_clips),
                        "segment_duration": segment_duration,
                        "audio_results": audio_results,
                        "concat_file": concat_path,
                        "output_file": final_output_path,
                        "photographer_credits": [f"Video by {clip['video_data']['photographer']} on Pexels" for clip in processed_clips]
                    }
                    results.append(multi_result)
                    logger.info(f"✅ Multi-clip reel created successfully: {final_output_path}")
                    
                else:
                    logger.error("❌ Failed to concatenate clips")
                    return []
            
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'single' or 'multi'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in convert_to_reel_with_audio: {e}")
            raise

    def convert_to_reel(self, query: str, per_page: int = 3) -> List[Dict]:
        """Convert videos to reel format using CrewAI workflow"""
        try:
            results = []
            
            # Task 1: Search for videos
            search_task = Task(
                description=f"""Search for {per_page} high-quality videos on Pexels using the query: "{query}". 
                Focus on videos that would work well for vertical social media content. Return detailed 
                information about each video including download URLs and dimensions.""",
                agent=self.search_agent,
                expected_output="JSON list of video information including IDs, dimensions, download URLs, and metadata"
            )
            
            # Execute search
            search_crew = Crew(
                agents=[self.search_agent],
                tasks=[search_task],
                process=Process.sequential,
                verbose=True
            )
            
            search_result = search_crew.kickoff()
            videos_data = json.loads(str(search_result))
            
            logger.info(f"Found {len(videos_data)} videos for processing")
            
            # Process each video
            for idx, video_data in enumerate(videos_data):
                try:
                    logger.info(f"Processing video {idx + 1}/{len(videos_data)}: ID {video_data['id']}")
                    
                    # Download video
                    video_filename = f"video_{video_data['id']}.mp4"
                    video_path = self.download_video(video_data['download_url'], video_filename)
                    
                    # Task 2: Video processing - save to unaudio_video first for consistency
                    unaudio_filename = f"reel_{video_data['id']}_{query.replace(' ', '_')}_unaudio.mp4"
                    unaudio_dir = os.path.join(self.output_dir, "unaudio_video")
                    unaudio_output_path = os.path.join(unaudio_dir, unaudio_filename)
                    
                    # Ensure unaudio_video directory exists
                    os.makedirs(unaudio_dir, exist_ok=True)
                    
                    processing_task = Task(
                        description=f"""Process the video at {video_path} by scaling it to exactly 720x1280 pixels (9:16 aspect ratio). 
                        Maintain aspect ratio with padding if needed. Preserve audio quality and optimize for social media playback. 
                        Save to {unaudio_output_path}.""",
                        agent=self.processing_agent,
                        expected_output="JSON object with processing results and output file information"
                    )
                    
                    processing_crew = Crew(
                        agents=[self.processing_agent],
                        tasks=[processing_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    processing_result = processing_crew.kickoff()
                    processing_data = json.loads(str(processing_result))
                    
                    if processing_data.get("success"):
                        # Log that the scaled video has been saved
                        logger.info(f"✅ SCALED VIDEO SAVED: {unaudio_output_path}")
                        logger.info(f"🎬 You can now check the video quality at: {unaudio_output_path}")
                        
                        # For no-audio mode, copy unaudio to final output
                        final_output_filename = f"reel_{video_data['id']}_{query.replace(' ', '_')}.mp4"
                        final_output_path = os.path.join(self.output_dir, final_output_filename)
                        import shutil
                        shutil.copy2(unaudio_output_path, final_output_path)
                        
                        result = {
                            "original_video": video_data,
                            "processing_result": processing_data,
                            "unaudio_file": unaudio_output_path,
                            "output_file": final_output_path,
                            "photographer_credit": f"Video by {video_data['photographer']} on Pexels"
                        }
                        results.append(result)
                        logger.info(f"Successfully processed video {video_data['id']}")
                    else:
                        logger.error(f"Failed to process video {video_data['id']}")
                        
                except Exception as e:
                    logger.error(f"Error processing video {video_data.get('id', 'unknown')}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in convert_to_reel: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

def main():
    """Example usage of the Video Reel Converter with both single and multi modes"""
    
    # Pexels API key
    PEXELS_API_KEY = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
    
    # Check if FFmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg to use this tool.")
        return
    
    # Initialize converter
    converter = VideoReelConverter(PEXELS_API_KEY)
    
    try:
        # Demo both modes
        query = "ocean waves"
        
        print("\n" + "="*80)
        print("🎬 VIDEO REEL CONVERTER - SINGLE vs MULTI MODE DEMO")
        print("="*80)
        
        # EXAMPLE 1: Single Mode (Default behavior)
        print(f"\n📱 SINGLE MODE: Converting individual videos for '{query}'")
        print("-" * 60)
        
        audio_options_single = {
            "music": True,
            "music_style": "calm peaceful ambient"
        }
        
        single_results = converter.convert_to_reel_with_audio(
            query=query, 
            audio_options=audio_options_single, 
            per_page=2, 
            mode='single'  # Default mode
        )
        
        print(f"✅ Single mode completed: {len(single_results)} individual reels created")
        for i, result in enumerate(single_results, 1):
            print(f"  Reel {i}: {result['output_file']}")
            print(f"    - Original: {result['original_video']['width']}x{result['original_video']['height']}")
            print(f"    - Photographer: {result['photographer_credit']}")
        
        # EXAMPLE 2: Multi Mode (New feature)
        print(f"\n🎞️  MULTI MODE: Creating dynamic multi-clip reel for '{query}'")
        print("-" * 60)
        
        audio_options_multi = {
            "music": True,
            "music_style": "cinematic epic upbeat",
            "voice": True,
            "voice_text": "Experience the power and beauty of the ocean through these stunning visuals"
        }
        
        multi_results = converter.convert_to_reel_with_audio(
            query=query, 
            audio_options=audio_options_multi, 
            per_page=6,  # Will fetch 6 videos for segments  
            mode='multi'  # New multi-clip mode
        )
        
        print(f"✅ Multi mode completed: {len(multi_results)} multi-clip reels created")
        for i, result in enumerate(multi_results, 1):
            print(f"  Multi-Reel {i}: {result['output_file']}")
            print(f"    - Videos used: {result['segment_count']} clips")
            print(f"    - Segment duration: {result['segment_duration']} seconds each")
            print(f"    - Total sources: {len(result['videos_used'])} videos")
            print(f"    - Credits: {len(result['photographer_credits'])} photographers")
        
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        print("🔹 SINGLE MODE:")
        print("  ✓ Creates individual reels from separate videos")
        print("  ✓ Best for showcasing specific content")
        print("  ✓ Maintains original video pacing")
        print("  ✓ Good for longer-form content")
        
        print("\n🔹 MULTI MODE:")
        print("  ✓ Creates dynamic multi-clip reels")
        print("  ✓ Perfect for social media engagement")
        print("  ✓ Fast-paced, attention-grabbing")
        print("  ✓ Ideal for Instagram Reels & TikTok")
        print("  ✓ Automatic 3-4 second segments")
        print("  ✓ Seamless clip transitions")
        
        print(f"\n📁 All outputs saved to: {converter.output_dir}")
        print("\n💡 Usage Tips:")
        print("  - Use single mode for detailed content showcase")
        print("  - Use multi mode for viral, fast-paced content")
        print("  - Multi mode automatically selects best video segments")
        print("  - Both modes support background music + voice narration")
        print("  - Always credit photographers when using content")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        converter.cleanup()

def demo_single_mode():
    """Demo function showing single mode usage"""
    PEXELS_API_KEY = "YOUR_PEXELS_API_KEY"
    converter = VideoReelConverter(PEXELS_API_KEY)
    
    try:
        # Single mode example
        results = converter.convert_to_reel_with_audio(
            query="sunset beach",
            audio_options={
                "music": True,
                "music_style": "peaceful ambient relaxing"
            },
            per_page=3,
            mode='single'  # Process 3 individual videos
        )
        print(f"Created {len(results)} individual reels")
        
    finally:
        converter.cleanup()

def demo_multi_mode():
    """Demo function showing multi mode usage"""
    PEXELS_API_KEY = "YOUR_PEXELS_API_KEY" 
    converter = VideoReelConverter(PEXELS_API_KEY)
    
    try:
        # Multi mode example
        results = converter.convert_to_reel_with_audio(
            query="city nightlife",
            audio_options={
                "music": True,
                "music_style": "energetic electronic upbeat",
                "voice": True,
                "voice_text": "Discover the vibrant energy of city nightlife"
            },
            per_page=6,  # Fetch 6 videos to create segments
            mode='multi'  # Create one dynamic multi-clip reel
        )
        print(f"Created {len(results)} multi-clip reel with dynamic segments")
        
    finally:
        converter.cleanup()

if __name__ == "__main__":
    main()