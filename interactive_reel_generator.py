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
    custom_query: Optional[str] = None

class ReelGeneratorUI:
    """Interactive Reel Generator with User Interface"""
    
    def __init__(self):
        self.pexels_api_key = "D5KPwqY6nRIZIkM93E2Hc7mQowQOAdBIIBgPDQUqm2iNeJosigMOTG4t"
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
        
        print(f"🎭 Mood: {mood_name}")
        print(f"⏱️  Duration: {duration_name}")
        print(f"🎨 Style: {style_name}")
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
            
            # Create preferences object
            preferences = UserPreferences(
                category=category_choice,
                subcategory=subcategory_choice,
                mood=mood_choice,
                duration_preference=duration_choice,
                style=style_choice,
                custom_query=custom_search
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
            
            # Initialize converter
            converter = VideoReelConverter(self.pexels_api_key)
            
            try:
                # Search for videos (get more options)
                print(f"🔍 Searching for: '{search_query}'")
                
                # First get the search results without processing
                search_crew = converter._setup_search_crew(search_query, per_page=5)
                search_result = search_crew.kickoff()
                videos_data = json.loads(str(search_result))
                
                if not videos_data:
                    print("❌ No videos found. Try different specifications.")
                    return
                
                # Filter by duration preference
                duration_filter = self.duration_preferences[preferences.duration_preference]['filter']
                filtered_videos = self.filter_videos_by_duration(videos_data, duration_filter)
                
                # Auto-select best videos
                selected = self.auto_select_videos(filtered_videos)
                if not selected:
                    print("👋 No suitable videos found!")
                    return
                
                # Convert selected videos
                videos_to_convert = selected['videos']
                print(f"\n🎬 Converting {len(videos_to_convert)} video(s)...")
                
                results = []
                for i, video_data in enumerate(videos_to_convert, 1):
                    print(f"\n📹 Processing video {i}/{len(videos_to_convert)}: {video_data['id']}")
                    result = converter._process_single_video(video_data, search_query)
                    if result:
                        results.append(result)
                
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
            print(f"\n📹 Video {i}:")
            print(f"  📄 File: {os.path.basename(result['output_file'])}")
            print(f"  📐 Resolution: {result['processing_result']['final_resolution']}")
            print(f"  🎭 Original by: {result['photographer_credit']}")
            print(f"  🎯 Smart cropping: {'Yes' if result['detection_result']['roi_detected'] else 'Center crop'}")
            
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
        
        print("\n💡 Tips:")
        print("  • Always credit the original photographers")
        print("  • Videos are optimized for social media")
        print("  • Ready to upload directly!")
        print("\n" + "="*80)

def main():
    """Main function"""
    generator = ReelGeneratorUI()
    generator.run()

if __name__ == "__main__":
    main()