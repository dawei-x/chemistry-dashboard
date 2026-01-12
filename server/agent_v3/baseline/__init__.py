"""
Baseline Agent for AIED 2026 Comparison

This module provides a transcript-only baseline variant of Agent V3.
It uses the same LangGraph workflow but with restricted tools:
- No concept maps
- No 7C analysis
- No LIWC metrics
- Speaker data limited to raw transcript utterances

This enables fair comparison to demonstrate how heterogeneous artifacts
enhance LLM reasoning capabilities.
"""

from .graph import run_baseline_agent
from .routes import baseline_bp

__all__ = ['run_baseline_agent', 'baseline_bp']
