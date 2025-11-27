"""
Fetch available HeyGen avatars from your account
Run this to get valid avatar IDs for config.py
"""

import requests
from config import Config

def fetch_avatars():
    """Fetch all available avatars from HeyGen account"""

    if not Config.HEYGEN_API_KEY:
        print("❌ Error: HEYGEN_API_KEY not set in .env")
        return

    headers = {
        "X-Api-Key": Config.HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }

    print("\n🔍 Fetching avatars from HeyGen account...")
    print("="*60)

    try:
        # Fetch avatars list
        response = requests.get(
            f"{Config.HEYGEN_BASE_URL}/avatars",
            headers=headers,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        avatars = data.get("data", {}).get("avatars", [])

        if not avatars:
            print("❌ No avatars found in your account")
            return

        print(f"\n✅ Found {len(avatars)} avatars:\n")

        # Print avatars in Python list format for easy copy-paste
        print("HEYGEN_AVATARS = [")
        for avatar in avatars:
            avatar_id = avatar.get("avatar_id")
            avatar_name = avatar.get("avatar_name", "Unknown")
            gender = avatar.get("gender", "unknown")

            print(f'    {{"id": "{avatar_id}", "name": "{avatar_name}", "gender": "{gender}"}},')

        print("]")

        print("\n" + "="*60)
        print("✅ Copy the above list and replace HEYGEN_AVATARS in config.py")
        print("="*60)

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    fetch_avatars()
