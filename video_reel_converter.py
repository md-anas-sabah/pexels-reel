#!/usr/bin/env python3
"""
AI Video Reel Converter using CrewAI and Pexels API
Converts videos to 9:16 aspect ratio (720x1280) with smart object detection cropping
"""

import os
import cv2
import numpy as np
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

class VideoSearchInput(BaseModel):
    """Input model for video search"""
    query: str = Field(description="Search query for videos")
    per_page: int = Field(default=5, description="Number of videos to fetch")

class VideoProcessInput(BaseModel):
    """Input model for video processing"""
    video_path: str = Field(description="Path to the video file")
    output_path: str = Field(description="Output path for processed video")

class PexelsVideoSearchTool(BaseTool):
    """Tool for searching videos on Pexels"""
    name: str = "pexels_video_search"
    description: str = "Search for videos on Pexels using a query keyword"
    
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._base_url = "https://api.pexels.com/videos"
    
    def _run(self, query: str, per_page: int = 5) -> str:
        """Search for videos on Pexels"""
        try:
            headers = {"Authorization": self._api_key}
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": "landscape"  # Prefer landscape for smart cropping
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

class ObjectDetectionTool(BaseTool):
    """Tool for detecting objects and regions of interest in video frames"""
    name: str = "object_detection"
    description: str = "Detect faces, people, and objects in video frames to determine optimal cropping region"
    
    def __init__(self):
        super().__init__()
        self._face_cascade = None
        self._body_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """Load OpenCV cascade classifiers"""
        try:
            # Load face detection cascade
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Load upper body detection cascade
            body_cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            self._body_cascade = cv2.CascadeClassifier(body_cascade_path)
            
            logger.info("Object detection cascades loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cascades: {e}")
    
    def _run(self, video_path: str) -> str:
        """Analyze video frames to find optimal cropping region"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return f"Error: Could not open video file {video_path}"
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for analysis (every 10th frame)
            sample_interval = max(1, total_frames // 20)
            
            roi_candidates = []
            
            for frame_idx in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                ) if self._face_cascade is not None else []
                
                # Detect upper bodies
                bodies = self._body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
                ) if self._body_cascade is not None else []
                
                # Combine detections
                detections = list(faces) + list(bodies)
                
                if len(detections) > 0:
                    # Find bounding box of all detections
                    x_min = min([x for x, y, w, h in detections])
                    y_min = min([y for x, y, w, h in detections])
                    x_max = max([x + w for x, y, w, h in detections])
                    y_max = max([y + h for x, y, w, h in detections])
                    
                    # Add padding
                    padding_x = (x_max - x_min) * 0.2
                    padding_y = (y_max - y_min) * 0.2
                    
                    roi_x = max(0, int(x_min - padding_x))
                    roi_y = max(0, int(y_min - padding_y))
                    roi_w = min(frame_width - roi_x, int((x_max - x_min) + 2 * padding_x))
                    roi_h = min(frame_height - roi_y, int((y_max - y_min) + 2 * padding_y))
                    
                    roi_candidates.append((roi_x, roi_y, roi_w, roi_h))
            
            cap.release()
            
            if roi_candidates:
                # Find the most common ROI region
                roi_x = int(np.median([roi[0] for roi in roi_candidates]))
                roi_y = int(np.median([roi[1] for roi in roi_candidates]))
                roi_w = int(np.median([roi[2] for roi in roi_candidates]))
                roi_h = int(np.median([roi[3] for roi in roi_candidates]))
                
                # Ensure the ROI maintains a reasonable aspect ratio for 9:16 conversion
                target_aspect = 9.0 / 16.0
                current_aspect = roi_w / roi_h
                
                if current_aspect > target_aspect:
                    # Too wide, adjust width
                    new_width = int(roi_h * target_aspect)
                    roi_x = roi_x + (roi_w - new_width) // 2
                    roi_w = new_width
                else:
                    # Too tall, adjust height
                    new_height = int(roi_w / target_aspect)
                    roi_y = roi_y + (roi_h - new_height) // 2
                    roi_h = new_height
                
                # Ensure ROI is within frame bounds
                roi_x = max(0, min(roi_x, frame_width - roi_w))
                roi_y = max(0, min(roi_y, frame_height - roi_h))
                roi_w = min(roi_w, frame_width - roi_x)
                roi_h = min(roi_h, frame_height - roi_y)
                
                return json.dumps({
                    "roi_detected": True,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_width": roi_w,
                    "roi_height": roi_h,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "detections_count": len(roi_candidates)
                })
            else:
                # No objects detected, use center crop
                target_aspect = 9.0 / 16.0
                if frame_width / frame_height > target_aspect:
                    # Video is too wide
                    crop_width = int(frame_height * target_aspect)
                    crop_x = (frame_width - crop_width) // 2
                    return json.dumps({
                        "roi_detected": False,
                        "roi_x": crop_x,
                        "roi_y": 0,
                        "roi_width": crop_width,
                        "roi_height": frame_height,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "detections_count": 0
                    })
                else:
                    # Video is too tall or correct aspect
                    crop_height = int(frame_width / target_aspect)
                    crop_y = (frame_height - crop_height) // 2
                    return json.dumps({
                        "roi_detected": False,
                        "roi_x": 0,
                        "roi_y": crop_y,
                        "roi_width": frame_width,
                        "roi_height": crop_height,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "detections_count": 0
                    })
                    
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return f"Error in object detection: {str(e)}"

class VideoProcessingTool(BaseTool):
    """Tool for processing videos with smart cropping and resizing"""
    name: str = "video_processing"
    description: str = "Process video with smart cropping and resize to 720x1280 (9:16 aspect ratio)"
    
    def _run(self, video_path: str, output_path: str, roi_data: str) -> str:
        """Process video with FFmpeg using ROI data"""
        try:
            roi_info = json.loads(roi_data)
            
            # Extract ROI parameters - handle multiple formats
            if "roi_x" in roi_info:
                # Flat format from object detection tool
                crop_x = roi_info["roi_x"]
                crop_y = roi_info["roi_y"]
                crop_w = roi_info["roi_width"]
                crop_h = roi_info["roi_height"]
            elif "roi" in roi_info:
                # Nested format from agent processing
                crop_x = roi_info["roi"]["x"]
                crop_y = roi_info["roi"]["y"]
                crop_w = roi_info["roi"]["width"]
                crop_h = roi_info["roi"]["height"]
            elif "roi_coordinates" in roi_info:
                # Another nested format from agent processing
                crop_x = roi_info["roi_coordinates"]["x"]
                crop_y = roi_info["roi_coordinates"]["y"]
                crop_w = roi_info["roi_coordinates"]["width"]
                crop_h = roi_info["roi_coordinates"]["height"]
            else:
                return "Error: Invalid ROI data format"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # FFmpeg command for high-quality cropping and resizing with audio preservation
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=720:1280:flags=lanczos",
                "-c:a", "aac", "-b:a", "128k",  # High quality audio
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
                # Handle roi_detected from different formats
                roi_detected = roi_info.get("roi_detected", True)
                detections_count = roi_info.get("detections_count", 0)
                
                return json.dumps({
                    "success": True,
                    "output_path": output_path,
                    "crop_region": f"{crop_w}x{crop_h}+{crop_x}+{crop_y}",
                    "final_resolution": "720x1280",
                    "roi_detected": roi_detected,
                    "detections_count": detections_count
                })
            else:
                error_msg = result.stderr
                logger.error(f"FFmpeg error: {error_msg}")
                return f"Error processing video: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return f"Error in video processing: {str(e)}"

class VideoReelConverter:
    """Main class for converting videos to reel format using CrewAI"""
    
    def __init__(self, pexels_api_key: str):
        self.pexels_api_key = pexels_api_key
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = "output_reels"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tools
        self.pexels_tool = PexelsVideoSearchTool(pexels_api_key)
        self.detection_tool = ObjectDetectionTool()
        self.processing_tool = VideoProcessingTool()
        
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
        
        # Object Detection Agent
        self.detection_agent = Agent(
            role="Computer Vision Specialist",
            goal="Analyze videos to identify optimal cropping regions using object detection",
            backstory="""You are a computer vision expert specializing in object detection and 
            video analysis. You can identify faces, people, and important objects in videos to 
            determine the best cropping strategy for vertical video conversion.""",
            tools=[self.detection_tool],
            verbose=True
        )
        
        # Video Processing Agent
        self.processing_agent = Agent(
            role="Video Production Expert",
            goal="Convert videos to perfect reel format with smart cropping and optimal quality",
            backstory="""You are a video production specialist who creates high-quality vertical 
            videos for social media. You ensure videos are properly cropped, resized, and 
            optimized while maintaining audio quality.""",
            tools=[self.processing_tool],
            verbose=True
        )
    
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
                    
                    # Task 2: Object detection
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
                    
                    # Task 3: Video processing
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
                        result = {
                            "original_video": video_data,
                            "detection_result": json.loads(str(detection_result)),
                            "processing_result": processing_data,
                            "output_file": output_path,
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
            print(f"  Objects detected: {result['detection_result']['detections_count']}")
            print(f"  Smart cropping: {'Yes' if result['detection_result']['roi_detected'] else 'Center crop'}")
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