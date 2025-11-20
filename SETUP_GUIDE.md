# AI Reel Generation Agent - Setup Guide

## PHASE 1 COMPLETE ✅

Congratulations! Phase 1 (Project Restructuring & Foundation) is now complete.

### What's Been Done

1. ✅ **Project Directory Structure Created**
   - `services/` - API client services
   - `workflows/` - Workflow implementations
   - `utils/` - Utility modules
   - `output/` - Generated video output

2. ✅ **Configuration Files Updated**
   - `.env` - API keys and credentials (including Supabase)
   - `config.py` - Comprehensive configuration class with all settings
   - `requirements.txt` - All dependencies including Supabase

3. ✅ **Supabase File Uploader Implemented**
   - Production-ready file upload utility
   - Automatic signed URL generation
   - Fallback to temporary hosting for development

---

## Next Steps: Complete Phase 1 Setup

### Step 1: Install Dependencies

```bash
cd "/Users/anassabah/Downloads/Marqait/heygen+submagic "

# Activate virtual environment
source venv/bin/activate

# Install new dependencies
pip install -r requirements.txt
```

### Step 2: Set Up Supabase (REQUIRED for Production)

#### 2.1 Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign in or create an account
3. Click **"New Project"**
4. Choose organization and enter:
   - **Name**: `reel-generator` (or your preferred name)
   - **Database Password**: (generate a strong password)
   - **Region**: Choose closest to you
5. Click **"Create new project"** and wait ~2 minutes

#### 2.2 Get Supabase Credentials

1. In your Supabase project dashboard, click **⚙️ Project Settings** (bottom left)
2. Go to **API** section
3. Copy the following:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **Anon/Public Key** (starts with `eyJ...`)

#### 2.3 Create Storage Bucket

1. In Supabase dashboard, click **🗄️ Storage** in sidebar
2. Click **"New bucket"**
3. Enter:
   - **Name**: `reel-videos`
   - **Public bucket**: ❌ OFF (we'll use signed URLs)
   - **File size limit**: 100 MB
   - **Allowed MIME types**: Leave empty (or add: video/mp4, video/quicktime, video/x-msvideo)
4. Click **"Create bucket"**

#### 2.4 Update .env File

Open `.env` and update these lines:

```env
# Replace with your actual values
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...your-anon-key
SUPABASE_BUCKET=reel-videos
```

### Step 3: Test the Setup

Run this command to verify everything is working:

```bash
# Test configuration
python -c "from config import print_config_status; print_config_status()"

# Test file uploader
python -c "from utils.file_uploader import test_upload; test_upload()"
```

Expected output:
```
============================================================
AI REEL GENERATION AGENT - CONFIGURATION STATUS
============================================================

API Keys Status:
  ✓ Pexels API:    ✓ SET
  ✓ HeyGen API:    ✓ SET
  ✓ Submagic API:  ✓ SET
  ✓ Fal AI API:    ✓ SET
  ✓ Supabase:      ✓ SET

Output Directory: /Users/.../output
Target Resolution: 720x1280 (9:16)
Video Quality: CRF 18, Preset: slower

✅ Configuration is valid!
============================================================
```

---

## Current API Key Status

Based on your `.env` file:

- ✅ **Pexels API**: Configured
- ✅ **HeyGen API**: Configured (key name fixed: HEYGEN_API_KEY)
- ✅ **Submagic API**: Configured (key name fixed: SUBMAGIC_API_KEY)
- ✅ **Fal AI API**: Configured
- ⚠️ **Supabase**: **NEEDS SETUP** (placeholder values)

---

## Project Structure

```
heygen+submagic/
├── .env                        # API keys (✅ UPDATED)
├── config.py                   # Configuration (✅ UPDATED)
├── requirements.txt            # Dependencies (✅ UPDATED)
│
├── services/                   # ✅ NEW - API clients
│   ├── __init__.py
│   ├── pexels_service.py      # 🔜 Phase 2
│   ├── heygen_service.py      # 🔜 Phase 2
│   ├── submagic_service.py    # 🔜 Phase 2
│   └── audio_service.py       # 🔜 Phase 2
│
├── workflows/                  # ✅ NEW - Workflow logic
│   ├── __init__.py
│   ├── local_media_workflow.py    # 🔜 Phase 3
│   ├── pexels_workflow.py         # 🔜 Phase 3
│   └── heygen_workflow.py         # 🔜 Phase 3
│
├── utils/                      # ✅ NEW - Utilities
│   ├── __init__.py
│   ├── file_uploader.py       # ✅ DONE (Supabase)
│   └── video_processor.py     # 🔜 Phase 2
│
├── output/                     # ✅ NEW - Generated videos
│
├── video_reel_converter.py    # Existing Pexels converter
├── interactive_reel_generator.py  # Existing UI
└── main.py                     # Existing entry point
```

---

## What's Next: PHASE 2

Once setup is complete, we'll move to **PHASE 2: Core Service Implementation**:

1. **Pexels Service** - Refactor existing code into service module
2. **HeyGen Service** - Avatar video generation client
3. **Submagic Service** - Automated editing client
4. **Audio Service** - Fal AI integration (already working)
5. **Video Processor** - FFmpeg wrapper utility

---

## Troubleshooting

### Issue: Supabase import error

```bash
pip install supabase==2.3.0 storage3==0.8.0
```

### Issue: File upload fails

- Check Supabase credentials in `.env`
- Verify bucket name matches (`reel-videos`)
- Check bucket permissions in Supabase dashboard
- Test with: `python utils/file_uploader.py`

### Issue: FFmpeg not found

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Issue: API key errors

- Double-check all API keys in `.env`
- Ensure no extra spaces or quotes
- Run: `python -c "from config import print_config_status; print_config_status()"`

---

## Getting Help

- **Supabase Docs**: https://supabase.com/docs/guides/storage
- **HeyGen API**: https://docs.heygen.com
- **Submagic API**: https://submagic.co/api-docs
- **Pexels API**: https://www.pexels.com/api/documentation

---

## Ready to Continue?

Once you've completed the setup steps above, you'll be ready for **PHASE 2: Core Service Implementation**.

Run this to verify you're ready:

```bash
python -c "from config import print_config_status; exit(0 if print_config_status() else 1)"
```

If you see "✅ Configuration is valid!", you're good to go!

---

**Status**: PHASE 1 COMPLETE ✅
**Next**: PHASE 2 - Core Service Implementation 🔜
**Last Updated**: 2025-11-20
