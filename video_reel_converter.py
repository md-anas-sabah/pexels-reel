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

class AudioMixingTool(BaseTool):
    """Tool for mixing generated audio with video using FFmpeg"""
    name: str = "audio_mixing"
    description: str = "Mix background music and/or voice-over with video"
    
    def _run(self, video_path: str, audio_url: str, output_path: str) -> str:
        """Mix audio with video using FFmpeg"""
        try:
            # Download audio file
            temp_audio_path = f"/tmp/temp_audio_{os.getpid()}.mp3"
            
            # Download the audio
            response = requests.get(audio_url)
            with open(temp_audio_path, 'wb') as f:
                f.write(response.content)
            
            # Since we remove audio during video processing, video will have no audio
            # Just add the TTS audio to the silent video
            cmd = [
                "ffmpeg", "-i", video_path, "-i", temp_audio_path,
                "-c:v", "copy",  # Copy video without re-encoding
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",  # Match the duration of the shorter stream
                "-y", output_path
            ]
            
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
                    "message": "Audio mixed successfully"
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
            self.audio_mixing_tool = AudioMixingTool()
        else:
            self.music_tool = None
            self.tts_tool = None
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
                            mixing_result = audio_mixing_tool._run(
                                video_path=current_video_path,
                                audio_url=audio_data['audio_url'],
                                output_path=output_path
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

    def convert_to_reel_with_audio(self, query: str, audio_options: Dict = None, per_page: int = 3) -> List[Dict]:
        """Convert videos to reel format with optional audio generation"""
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
            
            # Process each video with audio options
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
    """Example usage of the Video Reel Converter"""
    
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
        # Example conversion
        query = "nature landscape"
        logger.info(f"Converting videos for query: '{query}'")
        
        results = converter.convert_to_reel(query, per_page=2)
        
        print("\n" + "="*80)
        print("CONVERSION RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\nVideo {i}:")
            print(f"  Original: {result['original_video']['width']}x{result['original_video']['height']}")
            print(f"  Duration: {result['original_video']['duration']} seconds")
            print(f"  Photographer: {result['photographer_credit']}")
            print(f"  Processing: {result['processing_result']['processing_method']}")
            print(f"  Output file: {result['output_file']}")
            print(f"  Final resolution: {result['processing_result']['final_resolution']}")
        
        print(f"\n✅ Successfully converted {len(results)} videos to reel format!")
        print(f"📁 Output directory: {converter.output_dir}")
        print("\n💡 Tips:")
        print("  - Videos are optimized for Instagram Reels and TikTok")
        print("  - Audio is preserved in AAC format")
        print("  - Files are optimized for fast streaming")
        print("  - Always credit photographers when using their content")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        converter.cleanup()

if __name__ == "__main__":
    main()