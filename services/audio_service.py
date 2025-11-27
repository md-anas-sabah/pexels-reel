"""
Audio Service for Fal AI
Handles music generation and text-to-speech using Fal AI APIs
"""

import os
import logging
from typing import Optional, Dict
from pathlib import Path
import fal_client
from config import Config


logger = logging.getLogger(__name__)


class AudioService:
    """Service for generating music and speech using Fal AI"""

    def __init__(self, fal_key: Optional[str] = None):
        """
        Initialize Audio service

        Args:
            fal_key: Fal AI API key (defaults to Config.FAL_AI_API_KEY)
        """
        self.fal_key = fal_key or Config.FAL_AI_API_KEY

        if not self.fal_key:
            raise ValueError("Fal AI API key is required")

        # Configure fal_client
        os.environ['FAL_KEY'] = self.fal_key

    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        prompt_strength: float = 2.0,
        balance_strength: float = 0.7
    ) -> str:
        """
        Generate background music using Sonauto V2.2

        Args:
            prompt: Text description of desired music style
            duration: Duration in seconds (default: 30)
            prompt_strength: How closely to follow prompt (>= 1.4, default: 2.0)
            balance_strength: Balance between elements (default: 0.7)

        Returns:
            URL to generated audio file
        """
        print(f"\n🎵 Generating music...")
        print(f"   Prompt: '{prompt}'")
        print(f"   Duration: {duration}s")

        try:
            result = fal_client.subscribe(
                "sonauto/v2/text-to-music",
                arguments={
                    "prompt": prompt,
                    "prompt_strength": prompt_strength,
                    "balance_strength": balance_strength,
                    "duration": duration
                }
            )

            audio_url = result.get("audio_url") or result.get("audio", {}).get("url")

            if not audio_url:
                raise Exception("No audio URL returned from Fal AI")

            print(f"   ✅ Music generated! URL: {audio_url}")
            return audio_url

        except Exception as e:
            logger.error(f"Music generation failed: {str(e)}")
            raise

    def text_to_speech(
        self,
        text: str,
        voice_id: str = "Custom",
        emotion: Optional[str] = None,
        language: str = "English",
        language_boost: bool = False
    ) -> str:
        """
        Generate speech from text using MiniMax 2.5 HD TTS

        Args:
            text: Text to convert to speech
            voice_id: Voice ID (default: "Custom")
            emotion: Voice emotion (e.g., "happy", "sad", "neutral")
            language: Language of the text (default: "English")
            language_boost: Enable language-specific optimization (default: False)

        Returns:
            URL to generated speech audio file
        """
        print(f"\n🎙️ Generating speech...")
        print(f"   Text: '{text[:50]}...' ({len(text)} chars)")
        print(f"   Voice: {voice_id}")
        print(f"   Language: {language}")
        print(f"   Emotion: {emotion}")

        # Map our voice IDs to actual API voice IDs
        actual_voice_id = Config.VOICE_ID_API_MAPPING.get(voice_id, "Wise_Woman")

        # Build arguments
        arguments = {
            "text": text,
            "voice_id": actual_voice_id
        }

        # Add emotion if provided
        if emotion and emotion != "neutral":
            arguments["emotion"] = emotion
            print(f"   Using emotion: {emotion}")

        # Add language boost for non-English
        if language_boost and language.lower() != "english":
            arguments["language_boost"] = language
            print(f"   Using language_boost: {language}")

        try:
            print(f"   🚀 Calling MiniMax 2.5 HD API...")

            result = fal_client.subscribe(
                "fal-ai/minimax/preview/speech-2.5-hd",
                arguments=arguments
            )

            audio_url = result.get("audio_url") or result.get("audio", {}).get("url")

            if not audio_url:
                raise Exception("No audio URL returned from TTS API")

            print(f"   ✅ Speech generated! URL: {audio_url}")
            return audio_url

        except Exception as e:
            logger.error(f"Text-to-speech failed: {str(e)}")
            raise

    def download_audio(self, audio_url: str, output_path: str) -> str:
        """
        Download audio file from URL

        Args:
            audio_url: URL of the audio file
            output_path: Local path to save audio

        Returns:
            Path to downloaded audio file
        """
        print(f"\n📥 Downloading audio...")
        print(f"   URL: {audio_url}")
        print(f"   Output: {output_path}")

        try:
            import requests

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(audio_url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"   Progress: {progress:.1f}%", end='\r')

            print(f"\n   ✅ Download complete! Size: {downloaded / 1024:.2f} KB")
            return output_path

        except Exception as e:
            logger.error(f"Audio download failed: {str(e)}")
            raise

    def generate_and_download_music(
        self,
        prompt: str,
        output_path: str,
        duration: int = 30
    ) -> str:
        """
        Convenience method: Generate music and download it

        Args:
            prompt: Music style description
            output_path: Local path to save audio
            duration: Duration in seconds

        Returns:
            Path to downloaded audio file
        """
        audio_url = self.generate_music(prompt, duration)
        return self.download_audio(audio_url, output_path)

    def generate_and_download_speech(
        self,
        text: str,
        output_path: str,
        voice_id: str = "Custom",
        emotion: Optional[str] = None,
        language: str = "English"
    ) -> str:
        """
        Convenience method: Generate speech and download it

        Args:
            text: Text to speak
            output_path: Local path to save audio
            voice_id: Voice ID
            emotion: Voice emotion
            language: Language of text

        Returns:
            Path to downloaded audio file
        """
        audio_url = self.text_to_speech(text, voice_id, emotion, language)
        return self.download_audio(audio_url, output_path)

    def generate_subtitles(self, audio_url: str) -> Dict:
        """
        Generate subtitles from audio using Fal AI Whisper

        Args:
            audio_url: Public URL to audio file

        Returns:
            Dict with subtitle data (word-level timestamps)
        """
        print(f"\n📝 Generating subtitles from audio...")
        print(f"   Audio URL: {audio_url}")

        try:
            result = fal_client.subscribe(
                "fal-ai/whisper",
                arguments={
                    "audio_url": audio_url,
                    "task": "transcribe",
                    "chunk_level": "word",  # Word-level timestamps
                    "version": "3"
                }
            )

            print(f"   ✅ Subtitles generated!")
            return result

        except Exception as e:
            logger.error(f"Subtitle generation failed: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Test Audio service
    service = AudioService()

    # Generate music
    music_url = service.generate_music(
        prompt="Calm and peaceful background music with soft piano",
        duration=15
    )
    print(f"Music URL: {music_url}")

    # Generate speech
    speech_url = service.text_to_speech(
        text="Hello! This is a test of the text-to-speech system.",
        voice_id="Calm_Women",
        emotion="happy",
        language="English"
    )
    print(f"Speech URL: {speech_url}")

    # Download audio
    service.download_audio(music_url, "output/test_music.mp3")
    service.download_audio(speech_url, "output/test_speech.mp3")
