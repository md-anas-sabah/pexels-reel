"""
AI Agent Service - Two-Stage Decision Engine
1. Claude API: Refines and enhances user's video idea
2. GPT-3.5-turbo: Makes structured decisions for video generation
"""

import json
from typing import Dict
from anthropic import Anthropic
from openai import OpenAI
from langdetect import detect
from config import Config


class AIAgentService:
    """AI-powered decision engine for video reel generation"""

    def __init__(self):
        # Initialize Anthropic client (Claude API)
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.anthropic = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        # Initialize OpenAI client (GPT-3.5)
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai = OpenAI(api_key=Config.OPENAI_API_KEY)

    def refine_idea_with_claude(self, user_idea: str) -> str:
        """
        Stage 1: Use Claude to refine and enhance the user's video idea

        Args:
            user_idea: Raw user input (e.g., "calming morning coffee video")

        Returns:
            Enhanced, detailed prompt optimized for video generation
        """

        system_prompt = """You are a professional video producer and creative director.
Your job is to take a user's rough video idea and refine it into a clear, detailed creative brief
that captures the essence, mood, visual style, and narrative flow.

CRITICAL: If the user writes in Hinglish (Hindi-English mixed), PRESERVE their exact words and spellings.
- Do NOT auto-correct Hinglish words (e.g., keep "wiwaha" as "wiwaha", NOT "woke" or "wedding")
- Do NOT translate Hinglish to pure English or pure Hindi
- RESPECT their language choice and word spellings exactly as written

Enhance the idea by:
- Identifying the core emotion/mood
- Suggesting visual elements and scenes
- Defining the target audience
- Clarifying the message/story
- Recommending tone and pacing

Keep your response focused and under 100 words."""

        user_prompt = f"""User's video idea: "{user_idea}"

Refine this into a detailed creative brief for a social media reel (15-30 seconds).
Focus on visual storytelling, mood, and engagement."""

        # Try multiple Claude models in order of preference
        models_to_try = [
            "claude-3-5-sonnet-20241022",  # Latest (Oct 2024)
            "claude-3-5-sonnet-20240620",  # June 2024
            "claude-3-sonnet-20240229"     # Feb 2024 fallback
        ]

        for model in models_to_try:
            try:
                message = self.anthropic.messages.create(
                    model=model,
                    max_tokens=300,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )

                refined_idea = message.content[0].text.strip()
                print(f"\n🎨 Claude-refined idea (using {model}):")
                print(f"   {refined_idea}")
                return refined_idea

            except Exception as e:
                if "not_found_error" in str(e) and model != models_to_try[-1]:
                    continue  # Try next model
                elif model == models_to_try[-1]:
                    print(f"⚠️  Claude refinement failed (all models unavailable)")
                    print("   Using original idea...")
                    return user_idea
                else:
                    print(f"⚠️  Claude error: {e}")
                    print("   Using original idea...")
                    return user_idea

        return user_idea

    def detect_language(self, text: str) -> str:
        """
        Detect language of the input text

        Detects Hinglish (mixed Hindi-English) and preserves it

        Args:
            text: User's input text

        Returns:
            Language code: "English", "Hindi", "Arabic", "Urdu", or "Hinglish"
        """
        try:
            # Check if text contains both English and Hindi/Devanagari characters
            has_english = any(ord(char) < 128 and char.isalpha() for char in text)
            has_hindi = any('\u0900' <= char <= '\u097F' for char in text)

            # If both present, it's Hinglish
            if has_english and has_hindi:
                print(f"   🌐 Detected: Hinglish (mixed Hindi-English)")
                return 'Hinglish'

            # Otherwise use langdetect
            detected = detect(text)

            language_map = {
                'en': 'English',
                'hi': 'Hindi',
                'ar': 'Arabic',
                'ur': 'Urdu'
            }

            detected_lang = language_map.get(detected, 'English')

            # If detected as English but contains transliterated Hindi words
            # (common Hinglish pattern), mark as Hinglish
            if detected_lang == 'English':
                hinglish_indicators = ['ki', 'ka', 'ke', 'hai', 'ho', 'ko', 'se', 'me', 'pe',
                                      'wala', 'wali', 'karo', 'karna', 'aur', 'ya', 'par']
                words = text.lower().split()
                if any(indicator in words for indicator in hinglish_indicators):
                    print(f"   🌐 Detected: Hinglish (transliterated Hindi-English)")
                    return 'Hinglish'

            return detected_lang

        except Exception as e:
            print(f"⚠️  Language detection failed: {e}, defaulting to English")
            return 'English'

    def analyze_idea(self, user_idea: str) -> Dict:
        """
        Stage 2: Use GPT-3.5-turbo to analyze refined idea and make decisions

        Args:
            user_idea: User's video idea (raw or refined)

        Returns:
            Dictionary with all video generation decisions:
            {
                "category": str,
                "keywords": [str],
                "duration": str,
                "music_style": str,
                "voice_style": str,
                "narration": str,
                "voice_id": str,
                "emotion": str,
                "language": str
            }
        """

        # Stage 1: Refine with Claude
        refined_idea = self.refine_idea_with_claude(user_idea)

        # Detect language
        language = self.detect_language(user_idea)

        # Stage 2: GPT-3.5 makes structured decisions
        system_prompt = f"""You are a professional video production AI assistant. Analyze the video idea and make optimal decisions for creating an engaging social media reel.

CRITICAL - HINGLISH SUPPORT:
- If the user writes in Hinglish (Hindi-English mixed), the narration script MUST preserve their exact words and spellings
- Do NOT auto-correct Hinglish words (e.g., "wiwaha" stays "wiwaha", NOT "woke", "vivah", or "wedding")
- Do NOT translate Hinglish to pure English or pure Hindi
- Keep the same language style and vocabulary the user used in their original idea
- Example: If user says "shaadi ki video", narration should use "shaadi" not "wedding" or "विवाह"
- Example: If user says "morning coffee routine", narration can use English freely
- Example: If user mixes both like "subah ka coffee routine", keep mixing both languages naturally

Return ONLY valid JSON with these exact fields:
{{
  "category": "Select ONE from: [Nature & Lifestyle, Urban & City Life, People & Activities, Abstract & Creative, Seasonal & Weather]",
  "keywords": ["3-5 Pexels search keywords"],
  "duration": "short/medium/long (short=5-15s, medium=15-30s, long=30+s)",
  "music_style": "Select ONE from: [Upbeat & Energetic, Calm & Peaceful, Cinematic & Epic, Corporate & Professional, Hip-Hop & Urban, Pop & Catchy]",
  "voice_style": "Select ONE from: [Professional Narrator, Friendly & Casual, Energetic & Excited, Calm & Soothing, Authoritative]",
  "narration": "Generate compelling 20-25 second TTS script in {language}. MUST be 65-80 words. CRITICAL: If user wrote in Hinglish, use the EXACT same words/spellings from their idea in the script (preserve Hinglish exactly as written). Make it engaging and complete.",
  "voice_id": "Select EXACTLY ONE from: [Wise_Woman, Friendly_Person, Inspirational_Girl, Deep_Voice_Man, Calm_Women, Casual_Guy, Lively_Girl, Patient_Man, Young_Knight, Determined_Man, Lovely_Girl, Decent_Boy] - DO NOT create new names",
  "emotion": "Select ONE from: [happy, sad, angry, fearful, surprised, disgusted, neutral]",
  "language": "{language}",
  "outro_text": "Generate a 4-5 word catchy outro/call-to-action text (e.g., 'Follow for more!', 'Visit us today!', 'Subscribe now!')"
}}

CRITICAL RULES - SCRIPT LENGTH:
- Target: 65-80 words for 20-25 second audio
- Minimum: 65 words (ensure full content)
- Maximum: 80 words (prevent truncation)
- Quality and completeness - make it engaging

CRITICAL RULES - VOICE ID:
- ONLY use these EXACT voice_id values: Wise_Woman, Friendly_Person, Inspirational_Girl, Deep_Voice_Man, Calm_Women, Casual_Guy, Lively_Girl, Patient_Man, Young_Knight, Determined_Man, Lovely_Girl, Decent_Boy
- DO NOT create custom voice names like "Energetic_Girl" or "Happy_Man"
- DO NOT combine words differently
- Copy the exact spelling from the list above

VOICE ID SUGGESTIONS:
- For energetic content: Lively_Girl, Casual_Guy
- For calm content: Calm_Women, Wise_Woman, Patient_Man
- For motivational: Inspirational_Girl, Determined_Man, Young_Knight
- For friendly: Friendly_Person, Lovely_Girl, Decent_Boy
- For authoritative: Deep_Voice_Man, Wise_Woman

OTHER RULES:
- Narration must be in {language}
- HINGLISH PRESERVATION: If the original user idea contains Hinglish words (e.g., "wiwaha", "shaadi", "subah"), the narration MUST use those EXACT same spellings. Do NOT auto-correct or translate them.
- Narration MUST be 65-80 words (approximately 20-25 seconds when spoken)
- Do NOT go below 65 words or above 80 words
- Match emotion and voice to the video mood
- Keywords should be Pexels-friendly search terms
- Return ONLY the JSON object, no additional text

IMPORTANT - SCRIPT LENGTH (CRITICAL):
- Minimum: 65 words (enforce strictly)
- Target: 70-75 words (ideal)
- Maximum: 80 words (strict limit)
- This ensures 20-25 seconds of engaging audio content"""

        user_prompt = f"""Original User Input: "{user_idea}"
Refined Creative Brief: "{refined_idea}"

IMPORTANT: When generating the narration script, preserve any Hinglish words from the ORIGINAL USER INPUT exactly as the user spelled them. Do not auto-correct or translate these words.

Analyze this and generate the decision JSON."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            # Extract JSON from response
            response_text = response.choices[0].message.content.strip()

            # Try to parse JSON
            # Sometimes GPT wraps JSON in ```json blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            decisions = json.loads(response_text)

            # Validate required fields
            required_fields = ["category", "keywords", "duration", "music_style",
                             "voice_style", "narration", "voice_id", "emotion", "language", "outro_text"]

            for field in required_fields:
                if field not in decisions:
                    raise ValueError(f"Missing required field: {field}")

            # Display decisions
            print("\n🤖 GPT-3.5 Decisions:")
            print(f"   📂 Category: {decisions['category']}")
            print(f"   🔍 Keywords: {', '.join(decisions['keywords'])}")
            print(f"   ⏱️  Duration: {decisions['duration']}")
            print(f"   🎵 Music: {decisions['music_style']}")
            print(f"   🎤 Voice: {decisions['voice_id']} ({decisions['voice_style']})")
            print(f"   🎭 Emotion: {decisions['emotion']}")
            print(f"   🌍 Language: {decisions['language']}")
            print(f"   📝 Narration: {decisions['narration'][:80]}...")
            print(f"   🎬 Outro: \"{decisions['outro_text']}\"")

            return decisions

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse GPT-3.5 response as JSON: {e}")
            print(f"   Raw response: {response_text[:200]}")
            raise

        except Exception as e:
            print(f"❌ GPT-3.5 analysis failed: {e}")
            raise

    def validate_decisions(self, decisions: Dict) -> bool:
        """
        Validate that AI decisions are within allowed options

        Args:
            decisions: AI-generated decisions dictionary

        Returns:
            True if valid, raises ValueError if invalid
        """

        # Validate category
        if decisions["category"] not in Config.VIDEO_CATEGORIES:
            raise ValueError(f"Invalid category: {decisions['category']}")

        # Validate duration
        if decisions["duration"] not in ["short", "medium", "long"]:
            raise ValueError(f"Invalid duration: {decisions['duration']}")

        # Validate music style
        if decisions["music_style"] not in Config.MUSIC_STYLES:
            raise ValueError(f"Invalid music style: {decisions['music_style']}")

        # Validate voice style
        if decisions["voice_style"] not in Config.VOICE_STYLES:
            raise ValueError(f"Invalid voice style: {decisions['voice_style']}")

        # Validate voice ID
        if decisions["voice_id"] not in Config.VOICE_IDS:
            raise ValueError(f"Invalid voice ID: {decisions['voice_id']}")

        # Validate and auto-correct emotion
        if decisions["emotion"] not in Config.EMOTIONS:
            # Map common emotions to valid API emotions
            emotion_mapping = {
                "determined": "neutral",
                "excited": "happy",
                "calm": "neutral",
                "confident": "neutral",
                "inspired": "happy",
                "motivated": "happy",
                "energetic": "happy",
                "peaceful": "neutral",
                "anxious": "fearful",
                "nervous": "fearful",
                "joyful": "happy",
                "melancholic": "sad",
                "aggressive": "angry",
                "shocked": "surprised",
                "amazed": "surprised"
            }

            original_emotion = decisions["emotion"]
            if original_emotion.lower() in emotion_mapping:
                decisions["emotion"] = emotion_mapping[original_emotion.lower()]
                print(f"   ℹ️  Mapped emotion '{original_emotion}' → '{decisions['emotion']}'")
            else:
                # Default to neutral if no mapping found
                print(f"   ⚠️  Unknown emotion '{original_emotion}', defaulting to 'neutral'")
                decisions["emotion"] = "neutral"

        # Validate language
        if decisions["language"] not in Config.LANGUAGES:
            raise ValueError(f"Invalid language: {decisions['language']}")

        return True


# Example usage
if __name__ == "__main__":
    # Test the AI agent
    agent = AIAgentService()

    test_ideas = [
        "I want a calming video about morning coffee routines",
        "Create an energetic workout motivation reel",
        "मुझे सुबह की योग प्रैक्टिस के बारे में एक वीडियो चाहिए"
    ]

    for idea in test_ideas:
        print(f"\n{'='*60}")
        print(f"Testing: {idea}")
        print('='*60)

        try:
            decisions = agent.analyze_idea(idea)
            agent.validate_decisions(decisions)
            print("\n✅ Decisions validated successfully!")

        except Exception as e:
            print(f"\n❌ Error: {e}")
