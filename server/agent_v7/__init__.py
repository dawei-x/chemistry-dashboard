"""
BLINC Agent V7 - Full Context Agent

Based on Agent V3 with ALL truncation limits removed.
The LLM receives FULL transcripts, complete concept maps,
and detailed 7C analysis with all coded segments.

Key changes from V3:
1. No truncation of transcript utterances (was limited to 8)
2. No truncation of utterance text (was limited to 150 chars)
3. No truncation of concept map nodes (was limited to 8)
4. Full 7C coded segments included (was limited to 3 per dimension)
5. Increased max_tokens for synthesis output (4096 vs 2500)

This agent is designed to match ChatGPT-level analysis quality
by providing the LLM with complete data access.
"""

from .graph import create_agent_graph, get_compiled_graph
from .routes import agent_v7_bp

__all__ = ['create_agent_graph', 'get_compiled_graph', 'agent_v7_bp']
