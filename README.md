# 🎬 AI Video Reel Generator

Complete AI-powered system for creating professional Instagram Reels, TikTok videos, and YouTube Shorts. Transforms any landscape video into perfect 9:16 format with intelligent object detection and smart cropping.

## ⚡ Quick Start

```bash
# Run the complete system
python main.py

# Or use the simple launcher
python run.py

# Quick command-line mode
python main.py --mode quick --query "sunset beach" --count 3
```

## Features

- 🔍 **Smart Object Detection**: Uses OpenCV face/body detection to identify important content
- 🎯 **Intelligent Cropping**: Avoids cutting out faces, people, and important objects
- 🤖 **CrewAI Agents**: Multi-agent system for video search, analysis, and processing
- 📱 **Reel-Ready Output**: Perfect 720x1280 resolution for social media
- 🎵 **Audio Preservation**: Maintains high-quality audio in AAC format
- 🚀 **Pexels Integration**: Search and download videos directly from Pexels API
- ⚡ **Optimized Output**: Fast-start MP4 files optimized for streaming

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** (required for video processing)

#### Install FFmpeg:

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from video_reel_converter import VideoReelConverter

# Initialize with your Pexels API key
converter = VideoReelConverter("YOUR_PEXELS_API_KEY")

# Convert videos for a specific query
results = converter.convert_to_reel("nature landscape", per_page=3)

# Print results
for result in results:
    print(f"Output: {result['output_file']}")
    print(f"Credit: {result['photographer_credit']}")

# Cleanup temporary files
converter.cleanup()
```

### Command Line Usage

```bash
python video_reel_converter.py
```

## How It Works

### 1. Video Search Agent
- Searches Pexels API for videos matching your query
- Filters for high-quality videos suitable for conversion
- Returns video metadata and download URLs

### 2. Object Detection Agent
- Analyzes video frames using OpenCV cascades
- Detects faces, people, and important objects
- Calculates optimal crop region with padding
- Falls back to center crop if no objects detected

### 3. Video Processing Agent
- Downloads videos from Pexels
- Applies smart cropping based on detection results
- Resizes to exact 720x1280 resolution
- Preserves audio quality and optimizes for streaming

## Output Format

- **Resolution**: 720x1280 (9:16 aspect ratio)
- **Video Codec**: H.264 (libx264)
- **Audio Codec**: AAC
- **Quality**: CRF 23 (high quality)
- **Optimization**: Fast-start enabled for streaming

## Configuration

### Pexels API Key
Get your free API key from [https://www.pexels.com/api/](https://www.pexels.com/api/)

### Object Detection Models
The system uses OpenCV's built-in Haar cascades:
- `haarcascade_frontalface_default.xml` - Face detection
- `haarcascade_upperbody.xml` - Upper body detection

## Example Results

```
Video 1:
  Original: 1920x1080
  Duration: 15 seconds
  Photographer: Video by John Doe on Pexels
  Objects detected: 3
  Smart cropping: Yes
  Output file: output_reels/reel_12345_nature_landscape.mp4
  Final resolution: 720x1280
```

## Error Handling

The system includes comprehensive error handling for:
- Network failures during video download
- Invalid video formats
- Object detection errors
- FFmpeg processing errors
- API rate limiting

## Attribution Requirements

When using Pexels content, always provide proper attribution:
- Link back to Pexels: "Videos provided by Pexels"
- Credit photographers: "Video by [Photographer Name] on Pexels"

## Performance Tips

1. **Video Duration**: Shorter videos (10-30 seconds) process faster
2. **Internet Speed**: Faster connection improves download times
3. **System Resources**: More RAM helps with video processing
4. **Storage**: Ensure adequate disk space for temporary files

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```
Error: FFmpeg is not installed or not in PATH
```
Solution: Install FFmpeg and ensure it's in your system PATH

**API Rate Limit:**
```
Error: Too Many Requests (429)
```
Solution: Wait before making more requests or upgrade your Pexels plan

**Object Detection Errors:**
```
Error loading cascades
```
Solution: Ensure OpenCV is properly installed with `pip install opencv-python`

## License

This project is for educational and non-commercial use. Always respect Pexels' terms of service and properly attribute content creators.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Pexels API documentation
3. Open an issue on GitHub