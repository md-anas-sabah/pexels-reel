#!/usr/bin/env python3
"""
Interactive Reel Generator with User Specifications
User ko specify karne dete hai ki kya type ka video chahiye
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from video_reel_converter import VideoReelConverter
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

@dataclass
class UserPreferences:
    """User ki video preferences"""
    category: str
    subcategory: str
    mood: str
    duration_preference: str
    style: str
    audio_option: str
    music_style: Optional[str] = None
    voice_style: Optional[str] = None
    voice_text: Optional[str] = None
    custom_query: Optional[str] = None
    mode: str = 'single'  # Add mode to preferences

class KeywordEnhancementAgent:
    """AI agent using OpenAI GPT-3.5 to enhance search keywords for better video results"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = None
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                print("🤖 AI Keyword Enhancement Agent initialized")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize OpenAI client: {e}")
                print("🔍 Will use original search terms without enhancement")
        else:
            print("⚠️  Warning: OPENAI_API_KEY not found")
            print("🔍 Will use original search terms without enhancement")
    
    def enhance_search_query(self, user_input: str, mood: str = "neutral", style: str = "general") -> List[str]:
        """
        Enhance user's abstract search term into specific, visual keywords
        
        Args:
            user_input: User's original search term (e.g., "motivational")
            mood: Video mood preference
            style: Video style preference
            
        Returns:
            List of enhanced, specific search keywords
        """
        if not self.client:
            # Fallback: return original query if OpenAI not available
            return [user_input]
        
        try:
            prompt = f"""
You are a video search optimization expert. Transform abstract search terms into specific, visual keywords that work well for stock video searches.

User Input: "{user_input}"
Mood: {mood}
Style: {style}

Generate 5-7 specific, visual search keywords that represent this concept. Focus on:
- Concrete, visual actions and scenes
- Things that can actually be filmed
- Diverse perspectives and scenarios
- Professional stock video content

Format your response as a JSON array of strings.

Examples:
- "motivational" → ["person climbing mountain", "athlete training workout", "business team celebrating success", "runner crossing finish line", "entrepreneur working late", "graduation ceremony celebration"]
- "peaceful" → ["calm lake reflection", "person meditating sunrise", "gentle waves beach", "quiet forest path", "zen garden stones", "slow motion falling leaves"]
- "technology" → ["hands typing laptop", "smartphone close up", "server room lights", "coding on multiple monitors", "robotic arm assembly", "digital network animation"]

Your response (JSON array only):
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a video search optimization expert. Always respond with valid JSON arrays of search keywords."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            # Parse the response
            enhanced_keywords = json.loads(response.choices[0].message.content.strip())
            
            if isinstance(enhanced_keywords, list) and len(enhanced_keywords) > 0:
                print(f"🤖 AI Enhanced '{user_input}' → {len(enhanced_keywords)} specific keywords")
                print(f"   Keywords: {', '.join(enhanced_keywords[:3])}...")
                return enhanced_keywords
            else:
                print(f"⚠️  AI response format issue, using original: {user_input}")
                return [user_input]
                
        except json.JSONDecodeError:
            print(f"⚠️  AI response parsing failed, using original: {user_input}")
            return [user_input]
        except Exception as e:
            print(f"⚠️  AI enhancement failed ({e}), using original: {user_input}")
            return [user_input]
    
    def select_best_keyword(self, enhanced_keywords: List[str], mode: str = 'single') -> str:
        """
        Select the best keyword from enhanced list based on mode
        
        Args:
            enhanced_keywords: List of AI-enhanced keywords
            mode: 'single' or 'multi' - affects keyword selection strategy
            
        Returns:
            Best keyword for the specified mode
        """
        if not enhanced_keywords:
            return "nature"  # Fallback
        
        if mode == 'multi':
            # For multi-clip mode, prefer broader terms that yield diverse results
            # Look for keywords that suggest variety
            broad_keywords = [kw for kw in enhanced_keywords if 
                            any(word in kw.lower() for word in ['people', 'various', 'different', 'team', 'group'])]
            if broad_keywords:
                return broad_keywords[0]
        
        # Default: return first (usually best) keyword
        return enhanced_keywords[0]

class ReelGeneratorUI:
    """Interactive Reel Generator with User Interface"""
    
    def __init__(self):
        self.pexels_api_key = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
        # Initialize AI keyword enhancement agent
        self.keyword_agent = KeywordEnhancementAgent()
        self.categories = {
            "1": {
                "name": "Nature & Lifestyle",
                "subcategories": {
                    "1": {"name": "Ocean & Beach", "queries": ["ocean waves", "beach sunset", "tropical beach", "sea shore"]},
                    "2": {"name": "Mountains & Hills", "queries": ["mountain landscape", "valley view", "peak sunset", "hill station"]},
                    "3": {"name": "Forest & Trees", "queries": ["forest aerial", "tree canopy", "jungle view", "woodland"]},
                    "4": {"name": "Sunrise & Sunset", "queries": ["golden hour", "sunrise timelapse", "sunset sky", "horizon view"]}
                }
            },
            "2": {
                "name": "Urban & City Life",
                "subcategories": {
                    "1": {"name": "City Skyline", "queries": ["city lights", "skyline night", "urban architecture", "downtown view"]},
                    "2": {"name": "Street Life", "queries": ["busy street", "traffic flow", "pedestrian crossing", "market scene"]},
                    "3": {"name": "Modern Architecture", "queries": ["glass building", "modern design", "architectural detail", "urban structure"]},
                    "4": {"name": "Transportation", "queries": ["metro train", "highway traffic", "airport view", "bridge aerial"]}
                }
            },
            "3": {
                "name": "People & Activities",
                "subcategories": {
                    "1": {"name": "Fitness & Sports", "queries": ["gym workout", "running exercise", "yoga session", "sports activity"]},
                    "2": {"name": "Food & Cooking", "queries": ["cooking process", "food preparation", "restaurant kitchen", "coffee making"]},
                    "3": {"name": "Work & Business", "queries": ["office work", "meeting room", "coworking space", "business presentation"]},
                    "4": {"name": "Lifestyle", "queries": ["morning routine", "reading book", "creative work", "relaxation"]}
                }
            },
            "4": {
                "name": "Abstract & Creative",
                "subcategories": {
                    "1": {"name": "Colors & Patterns", "queries": ["colorful abstract", "geometric patterns", "paint mixing", "liquid motion"]},
                    "2": {"name": "Light & Effects", "queries": ["light rays", "neon glow", "particle effects", "lens flare"]},
                    "3": {"name": "Textures", "queries": ["fabric texture", "water ripples", "smoke effect", "fire flames"]},
                    "4": {"name": "Minimalist", "queries": ["simple design", "clean aesthetic", "minimal movement", "geometric shapes"]}
                }
            },
            "5": {
                "name": "Seasonal & Weather",
                "subcategories": {
                    "1": {"name": "Spring", "queries": ["blooming flowers", "green leaves", "spring rain", "fresh nature"]},
                    "2": {"name": "Summer", "queries": ["bright sunshine", "summer beach", "blue sky", "vacation vibes"]},
                    "3": {"name": "Autumn", "queries": ["falling leaves", "autumn colors", "harvest season", "cozy atmosphere"]},
                    "4": {"name": "Winter", "queries": ["snow falling", "winter landscape", "frost patterns", "cold weather"]}
                }
            }
        }
        
        self.moods = {
            "1": {"name": "Energetic", "keywords": ["dynamic", "fast", "active", "vibrant"]},
            "2": {"name": "Calm & Peaceful", "keywords": ["serene", "peaceful", "calm", "tranquil"]},
            "3": {"name": "Inspiring", "keywords": ["motivational", "uplifting", "inspiring", "positive"]},
            "4": {"name": "Professional", "keywords": ["corporate", "business", "professional", "clean"]},
            "5": {"name": "Creative", "keywords": ["artistic", "creative", "unique", "innovative"]}
        }
        
        self.duration_preferences = {
            "1": {"name": "Short (5-15 seconds)", "filter": "short"},
            "2": {"name": "Medium (15-30 seconds)", "filter": "medium"},
            "3": {"name": "Long (30+ seconds)", "filter": "long"},
            "4": {"name": "Any Duration", "filter": "any"}
        }
        
        self.styles = {
            "1": {"name": "Professional/Corporate", "style": "professional"},
            "2": {"name": "Creative/Artistic", "style": "creative"},
            "3": {"name": "Social Media Ready", "style": "social"},
            "4": {"name": "Cinematic", "style": "cinematic"}
        }
        
        self.audio_options = {
            "1": {"name": "No Audio (Video only)", "option": "none"},
            "2": {"name": "Background Music only", "option": "music"},
            "3": {"name": "Voice Narration only", "option": "voice"}, 
            "4": {"name": "Music + Voice Narration", "option": "both"}
        }
        
        self.music_styles = {
            "1": {"name": "Upbeat & Energetic", "style": "upbeat energetic electronic"},
            "2": {"name": "Calm & Peaceful", "style": "calm peaceful ambient"},
            "3": {"name": "Cinematic & Epic", "style": "cinematic epic orchestral"},
            "4": {"name": "Corporate & Professional", "style": "corporate professional clean"},
            "5": {"name": "Hip-Hop & Urban", "style": "hip-hop urban beat"},
            "6": {"name": "Pop & Catchy", "style": "pop catchy uplifting"}
        }
        
        self.voice_styles = {
            "1": {"name": "Professional Narrator", "style": "professional"},
            "2": {"name": "Friendly & Casual", "style": "friendly"},
            "3": {"name": "Energetic & Excited", "style": "energetic"},
            "4": {"name": "Calm & Soothing", "style": "calm"},
            "5": {"name": "Authoritative", "style": "authoritative"}
        }

    def display_welcome(self):
        """Welcome message"""
        print("\n" + "="*80)
        print("🎬 AI REEL GENERATOR - PERSONALIZED VIDEO CREATOR")
        print("="*80)
        print("📱 Create perfect Instagram Reels, TikTok videos, and YouTube Shorts")
        print("🎯 Aap specify kar sakte hai exactly kaisi video chahiye!")
        print("="*80 + "\n")

    def display_categories(self):
        """Display video categories"""
        print("📂 VIDEO CATEGORIES:")
        print("-" * 50)
        for key, category in self.categories.items():
            print(f"{key}. {category['name']}")
        print("\n0. Custom Search (Apna keyword enter kariye)")

    def display_subcategories(self, category_key: str):
        """Display subcategories for selected category"""
        category = self.categories[category_key]
        print(f"\n📁 {category['name']} - SUBCATEGORIES:")
        print("-" * 50)
        for key, subcategory in category['subcategories'].items():
            print(f"{key}. {subcategory['name']}")

    def display_moods(self):
        """Display mood options"""
        print("\n🎭 VIDEO MOOD/VIBE:")
        print("-" * 30)
        for key, mood in self.moods.items():
            print(f"{key}. {mood['name']}")

    def display_duration_preferences(self):
        """Display duration preferences"""
        print("\n⏱️  VIDEO DURATION PREFERENCE:")
        print("-" * 35)
        for key, duration in self.duration_preferences.items():
            print(f"{key}. {duration['name']}")

    def display_styles(self):
        """Display style options"""
        print("\n🎨 VIDEO STYLE:")
        print("-" * 20)
        for key, style in self.styles.items():
            print(f"{key}. {style['name']}")

    def display_audio_options(self):
        """Display audio options"""
        print("\n🎵 AUDIO OPTIONS:")
        print("-" * 25)
        for key, audio in self.audio_options.items():
            print(f"{key}. {audio['name']}")

    def display_music_styles(self):
        """Display music style options"""
        print("\n🎼 MUSIC STYLE:")
        print("-" * 20)
        for key, music in self.music_styles.items():
            print(f"{key}. {music['name']}")

    def display_voice_styles(self):
        """Display voice style options"""
        print("\n🎤 VOICE STYLE:")
        print("-" * 20)
        for key, voice in self.voice_styles.items():
            print(f"{key}. {voice['name']}")

    def display_generation_modes(self):
        """Display video generation mode options"""
        print("\n🎬 VIDEO GENERATION MODE:")
        print("-" * 30)
        print("1. 📱 Single Mode - Individual videos (classic mode)")
        print("   • Creates separate reels from different videos")
        print("   • Best for showcasing specific content")
        print("   • Traditional social media format")
        print()
        print("2. 🎞️  Multi Mode - Dynamic multi-clip reel (new!)")
        print("   • Creates ONE reel from multiple video segments")
        print("   • 3-4 second clips automatically trimmed & combined")
        print("   • Perfect for viral, fast-paced content")
        print("   • Ideal for Instagram Reels & TikTok")

    def get_voice_text(self) -> str:
        """Get custom voice text from user"""
        print("\n📝 VOICE NARRATION TEXT:")
        print("-" * 30)
        print("Examples:")
        print("• 'Discover amazing nature views in this beautiful location'")
        print("• 'Transform your daily routine with these simple tips'")
        print("• 'Experience the future of technology today'")
        text = input("Enter your narration text: ").strip()
        return text if text else "Check out this amazing video content!"

    def get_user_input(self, prompt: str, valid_options: List[str]) -> str:
        """Get valid user input"""
        while True:
            choice = input(f"\n{prompt}: ").strip()
            if choice in valid_options:
                return choice
            print(f"❌ Invalid choice! Please select from: {', '.join(valid_options)}")

    def get_custom_query(self) -> str:
        """Get custom search query from user"""
        print("\n🔍 CUSTOM SEARCH:")
        print("-" * 20)
        print("Examples: 'coffee shop morning', 'dancing people', 'car driving night'")
        query = input("Enter your custom search query: ").strip()
        return query if query else "trending video"

    def generate_search_query(self, preferences: UserPreferences) -> str:
        """Generate optimized search query based on user preferences"""
        if preferences.custom_query:
            base_query = preferences.custom_query
        else:
            # Get base queries from category/subcategory
            category = self.categories[preferences.category]
            subcategory = category['subcategories'][preferences.subcategory]
            base_queries = subcategory['queries']
            
            # Select best query based on mood
            mood_keywords = self.moods[preferences.mood]['keywords']
            base_query = base_queries[0]  # Default to first
            
            # Add mood keyword to enhance search
            mood_keyword = mood_keywords[0]
            base_query = f"{base_query} {mood_keyword}"
        
        return base_query

    def display_preferences_summary(self, preferences: UserPreferences, search_query: str):
        """Display user's selected preferences"""
        print("\n" + "="*60)
        print("📋 AAPKI VIDEO SPECIFICATIONS:")
        print("="*60)
        
        if not preferences.custom_query:
            category_name = self.categories[preferences.category]['name']
            subcategory_name = self.categories[preferences.category]['subcategories'][preferences.subcategory]['name']
            print(f"📂 Category: {category_name}")
            print(f"📁 Subcategory: {subcategory_name}")
        else:
            print(f"🔍 Custom Search: {preferences.custom_query}")
            
        mood_name = self.moods[preferences.mood]['name']
        duration_name = self.duration_preferences[preferences.duration_preference]['name']
        style_name = self.styles[preferences.style]['name']
        audio_name = self.audio_options[preferences.audio_option]['name']
        
        print(f"🎭 Mood: {mood_name}")
        print(f"⏱️  Duration: {duration_name}")
        print(f"🎨 Style: {style_name}")
        print(f"🎵 Audio: {audio_name}")
        
        # Show specific audio preferences
        if preferences.music_style:
            music_name = self.music_styles[preferences.music_style]['name']
            print(f"🎼 Music Style: {music_name}")
        
        if preferences.voice_style:
            voice_name = self.voice_styles[preferences.voice_style]['name']
            print(f"🎤 Voice Style: {voice_name}")
            
        if preferences.voice_text:
            print(f"📝 Narration: '{preferences.voice_text[:50]}{'...' if len(preferences.voice_text) > 50 else ''}'")
        
        print(f"🔍 Search Query: '{search_query}'")
        print("="*60)

    def confirm_preferences(self) -> bool:
        """Confirm user preferences"""
        while True:
            choice = input("\n✅ Proceed with these specifications? (y/n): ").strip().lower()
            if choice in ['y', 'yes', 'हां', 'ha']:
                return True
            elif choice in ['n', 'no', 'नहीं', 'nahi']:
                return False
            print("Please enter 'y' for yes or 'n' for no")

    def collect_user_preferences(self) -> Optional[UserPreferences]:
        """Collect all user preferences"""
        try:
            # Display categories
            self.display_categories()
            valid_categories = list(self.categories.keys()) + ['0']
            category_choice = self.get_user_input(
                "Select category number", valid_categories
            )
            
            # Handle custom search
            if category_choice == '0':
                custom_query = self.get_custom_query()
                # Still need other preferences
                category_choice = '1'  # Default for other selections
                subcategory_choice = '1'
                custom_search = custom_query
            else:
                # Display subcategories
                self.display_subcategories(category_choice)
                valid_subcategories = list(self.categories[category_choice]['subcategories'].keys())
                subcategory_choice = self.get_user_input(
                    "Select subcategory number", valid_subcategories
                )
                custom_search = None
            
            # Get mood
            self.display_moods()
            valid_moods = list(self.moods.keys())
            mood_choice = self.get_user_input("Select mood number", valid_moods)
            
            # Get duration preference
            self.display_duration_preferences()
            valid_durations = list(self.duration_preferences.keys())
            duration_choice = self.get_user_input("Select duration preference", valid_durations)
            
            # Get style
            self.display_styles()
            valid_styles = list(self.styles.keys())
            style_choice = self.get_user_input("Select style number", valid_styles)
            
            # Get audio options
            self.display_audio_options()
            valid_audio = list(self.audio_options.keys())
            audio_choice = self.get_user_input("Select audio option", valid_audio)
            
            # Get audio-specific preferences
            music_style_choice = None
            voice_style_choice = None
            voice_text_input = None
            
            audio_option = self.audio_options[audio_choice]["option"]
            
            if audio_option in ["music", "both"]:
                self.display_music_styles()
                valid_music = list(self.music_styles.keys())
                music_style_choice = self.get_user_input("Select music style", valid_music)
            
            if audio_option in ["voice", "both"]:
                self.display_voice_styles()
                valid_voice = list(self.voice_styles.keys())
                voice_style_choice = self.get_user_input("Select voice style", valid_voice)
                voice_text_input = self.get_voice_text()
            
            # Get video generation mode
            self.display_generation_modes()
            valid_modes = ["1", "2"]
            mode_choice = self.get_user_input("Select generation mode", valid_modes)
            mode = "single" if mode_choice == "1" else "multi"
            
            # Create preferences object
            preferences = UserPreferences(
                category=category_choice,
                subcategory=subcategory_choice,
                mood=mood_choice,
                duration_preference=duration_choice,
                style=style_choice,
                audio_option=audio_choice,
                music_style=music_style_choice,
                voice_style=voice_style_choice,
                voice_text=voice_text_input,
                custom_query=custom_search,
                mode=mode
            )
            
            return preferences
            
        except KeyboardInterrupt:
            print("\n\n👋 Bye! Aage aake video banayenge!")
            return None
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None

    def filter_videos_by_duration(self, videos: List[Dict], duration_filter: str) -> List[Dict]:
        """Filter videos based on duration preference"""
        if duration_filter == "any":
            return videos
        
        filtered = []
        for video in videos:
            duration = video.get('duration', 0)
            
            if duration_filter == "short" and 5 <= duration <= 15:
                filtered.append(video)
            elif duration_filter == "medium" and 15 < duration <= 30:
                filtered.append(video)
            elif duration_filter == "long" and duration > 30:
                filtered.append(video)
        
        # If no videos match duration filter, return original list
        return filtered if filtered else videos

    def auto_select_videos(self, videos: List[Dict]) -> Optional[Dict]:
        """Automatically select best videos based on quality criteria"""
        if not videos:
            print("❌ No videos found matching your criteria")
            return None
        
        print("\n" + "="*80)
        print("🎥 FOUND VIDEOS - AI SELECTING BEST OPTIONS:")
        print("="*80)
        
        # Score videos based on multiple criteria
        scored_videos = []
        for video in videos:
            score = 0
            
            # Quality score based on resolution
            width = video.get('width', 0)
            height = video.get('height', 0)
            resolution_score = (width * height) / 2073600  # Normalize to 1920x1080
            score += min(resolution_score, 1.0) * 30
            
            # Duration score (prefer videos between 10-30 seconds)
            duration = video.get('duration', 0)
            if 10 <= duration <= 30:
                score += 25
            elif 5 <= duration < 10 or 30 < duration <= 45:
                score += 15
            else:
                score += 5
            
            # Aspect ratio score (prefer wider videos for better cropping)
            aspect_ratio = video.get('aspect_ratio', 1.0)
            if aspect_ratio >= 1.5:  # Landscape videos
                score += 20
            elif aspect_ratio >= 1.0:
                score += 10
            else:
                score += 5
            
            # Add some randomness to avoid always picking the same type
            import random
            score += random.uniform(0, 15)
            
            scored_videos.append((score, video))
        
        # Sort by score (highest first)
        scored_videos.sort(key=lambda x: x[0], reverse=True)
        
        # Select only the best video
        selected_videos = [video for score, video in scored_videos[:1]]
        
        print(f"\n🤖 AI Selected the best video:")
        for i, video in enumerate(selected_videos, 1):
            print(f"\n{i}. 📹 {video.get('photographer', 'Unknown')} ka video")
            print(f"   ⏱️  Duration: {video.get('duration', 0)} seconds")
            print(f"   📐 Size: {video.get('width', 0)}x{video.get('height', 0)}")
            print(f"   🔗 Preview: {video.get('url', 'N/A')}")
        
        print(f"\n🚀 Proceeding with automatic conversion...")
        
        return {'all': True, 'videos': selected_videos}

    def run(self):
        """Main function to run the interactive reel generator"""
        try:
            # Welcome
            self.display_welcome()
            
            # Collect user preferences
            preferences = self.collect_user_preferences()
            if not preferences:
                return
            
            # Generate search query
            search_query = self.generate_search_query(preferences)
            
            # Display summary
            self.display_preferences_summary(preferences, search_query)
            
            # Auto-proceed without confirmation
            print("\n🚀 Starting video search and conversion...")
            
            # Initialize converter with audio support
            fal_key = os.getenv('FAL_KEY')
            converter = VideoReelConverter(self.pexels_api_key, fal_key)
            
            # Prepare audio options for converter
            audio_options = None
            audio_option_type = self.audio_options[preferences.audio_option]["option"]
            
            if audio_option_type != "none":
                audio_options = {
                    "music": audio_option_type in ["music", "both"],
                    "voice": audio_option_type in ["voice", "both"]
                }
                
                if preferences.music_style:
                    audio_options["music_style"] = self.music_styles[preferences.music_style]["style"]
                
                if preferences.voice_style and preferences.voice_text:
                    audio_options["voice_style"] = self.voice_styles[preferences.voice_style]["style"]
                    audio_options["voice_text"] = preferences.voice_text
                
                print(f"🎵 Audio enabled: {audio_option_type}")
            else:
                print("🔇 No audio will be added")
            
            try:
                # AI-enhanced keyword optimization
                if preferences.custom_query:
                    print(f"🔍 Original search: '{search_query}'")
                    print("🤖 Enhancing keywords with AI...")
                    
                    enhanced_keywords = self.keyword_agent.enhance_search_query(
                        user_input=search_query,
                        mood=self.moods[preferences.mood]["name"].lower(),
                        style=self.styles[preferences.style]["name"].lower()
                    )
                    
                    # Select best keyword based on mode
                    final_search_query = self.keyword_agent.select_best_keyword(
                        enhanced_keywords, preferences.mode
                    )
                    
                    print(f"🎯 Final search query: '{final_search_query}'")
                else:
                    final_search_query = search_query
                    print(f"🔍 Using category-based search: '{final_search_query}'")
                
                # Determine per_page based on mode
                if preferences.mode == 'multi':
                    per_page = 6  # Fetch 6 videos for multi-clip segments
                    print(f"🎞️  Multi-clip mode: Fetching {per_page} videos for dynamic reel")
                else:
                    per_page = 3  # Fetch 3 videos for individual processing
                    print(f"📱 Single mode: Fetching {per_page} individual videos")
                
                # Use the enhanced conversion method
                results = converter.convert_to_reel_with_audio(
                    query=final_search_query, 
                    audio_options=audio_options, 
                    per_page=per_page,
                    mode=preferences.mode
                )
                
                # Display results
                self.display_results(results, preferences)
                
            finally:
                converter.cleanup()
                
        except KeyboardInterrupt:
            print("\n\n👋 Process cancelled. Bye!")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    def display_results(self, results: List[Dict], preferences: UserPreferences):
        """Display final results"""
        if not results:
            print("\n❌ No videos were successfully converted")
            return
        
        print("\n" + "="*80)
        print("🎉 VIDEO CONVERSION COMPLETED!")
        print("="*80)
        print(f"✅ Successfully converted {len(results)} video(s)")
        print(f"📁 Output folder: output_reels/")
        
        for i, result in enumerate(results, 1):
            if result.get('mode') == 'multi':
                # Multi-clip reel display
                print(f"\n🎞️  Multi-Clip Reel {i}:")
                print(f"  📄 File: {os.path.basename(result['output_file'])}")
                print(f"  📐 Resolution: 720x1280")
                print(f"  🎬 Clips used: {result['segment_count']} segments")
                print(f"  ⏱️  Segment duration: {result['segment_duration']} seconds each")
                print(f"  🎯 Processing: Multi-clip concatenation")
                
                # Show source videos
                print(f"  📺 Source videos: {len(result['videos_used'])}")
                for j, video in enumerate(result['videos_used'][:3], 1):  # Show first 3
                    print(f"     {j}. ID {video['id']} by {video['photographer']}")
                if len(result['videos_used']) > 3:
                    print(f"     ... and {len(result['videos_used']) - 3} more")
                
                # Show audio information if available
                if result.get('audio_results'):
                    audio_count = len(result['audio_results'])
                    print(f"  🎵 Audio tracks added: {audio_count}")
                    has_subtitles = False
                    for j, audio in enumerate(result['audio_results'], 1):
                        audio_type = audio.get('type', 'unknown')
                        print(f"     {j}. {audio_type.title()}")
                        if audio.get('srt_path') and audio_type == 'tts':
                            has_subtitles = True
                    
                    if has_subtitles:
                        print(f"  📝 Subtitles: Synchronized word-by-word")
                else:
                    print(f"  🔇 No audio added")
                    
            else:
                # Single video display
                print(f"\n📹 Video {i}:")
                print(f"  📄 File: {os.path.basename(result['output_file'])}")
                print(f"  📐 Resolution: {result['processing_result']['final_resolution']}")
                print(f"  🎭 Original by: {result['photographer_credit']}")
                print(f"  🎯 Processing: {result['processing_result'].get('processing_method', 'scale_with_padding')}")
                
                # Show audio information if available
                if result.get('audio_results'):
                    audio_count = len(result['audio_results'])
                    print(f"  🎵 Audio tracks added: {audio_count}")
                    has_subtitles = False
                    for j, audio in enumerate(result['audio_results'], 1):
                        audio_type = audio.get('type', 'unknown')
                        print(f"     {j}. {audio_type.title()}")
                        if audio.get('srt_path') and audio_type == 'tts':
                            has_subtitles = True
                    
                    if has_subtitles:
                        print(f"  📝 Subtitles: Synchronized word-by-word")
                else:
                    print(f"  🔇 No audio added")
            
            if os.path.exists(result['output_file']):
                file_size = os.path.getsize(result['output_file']) / (1024 * 1024)
                print(f"  💾 Size: {file_size:.1f} MB")
        
        print("\n📱 Ready for:")
        print("  • Instagram Reels")
        print("  • TikTok")
        print("  • YouTube Shorts")
        print("  • Facebook Reels")
        
        print(f"\n🎨 Style: {self.styles[preferences.style]['name']}")
        print(f"🎭 Mood: {self.moods[preferences.mood]['name']}")
        print(f"🎬 Mode: {preferences.mode.title()}")
        
        if preferences.mode == 'multi':
            print("\n🎞️  Multi-Clip Features:")
            print("  • Dynamic segment transitions")
            print("  • Perfect for viral content")
            print("  • Optimized for short attention spans")
            print("  • Professional concatenation")
        
        print("\n💡 Tips:")
        print("  • Always credit the original photographers")
        print("  • Videos are optimized for social media")
        print("  • Ready to upload directly!")
        if preferences.mode == 'multi':
            print("  • Multi-clip reels perform better on social media")
        print("\n" + "="*80)

def main():
    """Main function"""
    generator = ReelGeneratorUI()
    generator.run()

if __name__ == "__main__":
    main()