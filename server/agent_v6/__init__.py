"""
Agent V6: The Definitive Architecture

Embeds V3's domain expertise (operationalizations, epistemic hierarchy, triangulation)
into V4's simple ReAct loop architecture with first-class steering support.

Core Philosophy: "Encode domain expertise explicitly. Let the LLM reason with it freely."

Key Features:
- V3's analytical intelligence embedded in prompts
- V4's simple ReAct loop for speed
- First-class steering (prefer/exclude representations)
- Hypothesis testing protocol
- LLM reasoning beyond retrieval
"""

from .agent import run_agent, conversation_manager
from .routes import agent_v6_bp

__all__ = ['run_agent', 'conversation_manager', 'agent_v6_bp']
__version__ = '6.0.0'
