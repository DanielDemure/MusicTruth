"""
Researcher Agent.

Uses simple search or LLM knowledge to find context about an artist/track.
"""

from .client import LLMClient

class ResearcherAgent:
    """
    Finds background information about a track/artist.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    def research_context(self, artist: str, title: str) -> str:
        """
        Get context about the artist/track.
        
        In a full implementation, this might use a Search API (Google/DuckDuckGo).
        For now, we rely on the LLM's internal knowledge base.
        """
        if not self.llm.check_availability():
            return "LLM not available for research."
            
        system = """You are a Music Historian and Research Assistant. 
        Your goal is to provide brief, factual context about a musical artist and their typical production style.
        Focus on:
        1. Genre and Era.
        2. Known production techniques (e.g., lo-fi, autotune heavy, acoustic).
        3. Expectation of "Perfection" (e.g., is this a raw punk band or a polished pop star?).
        
        Keep it concise (under 200 words)."""
        
        prompt = f"Tell me about the production style of '{artist}', specifically for the track '{title}' if known. If unknown, describe the artist's general style."
        
        return self.llm.generate(system, prompt)
