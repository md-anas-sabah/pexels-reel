"""
Workflows package for AI Reel Generation Agent
Contains three main workflows:
1. Local Media Workflow - Process existing videos
2. Pexels Stock Workflow - AI-driven video generation from stock footage
3. HeyGen Avatar Workflow - AI avatar + B-roll compositing
"""

from .pexels_workflow import PexelsWorkflow

__all__ = ['PexelsWorkflow']
