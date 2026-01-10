"""
Critic and Reporter Agents.

Critic: Reviews technical metrics against context.
Reporter: Writes the final specific public report.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import LLMClient

class CriticAgent:
    """
    Analyzes technical findings through a critical lens.
    """
    
    def __init__(self, llm_client: 'LLMClient'):
        self.llm = llm_client
        
    def critique(self, analysis_results: Dict[str, Any], context: str) -> str:
        """
        Critically evaluate results.
        
        Args:
            analysis_results: Raw features and flags.
            context: Artist/Track background info.
        """
        if not self.llm.check_availability():
            return "LLM not available for critique."
            
        system = """You are a Senior Audio Forensic Analyst.
        Your job is to interpret technical audio analysis data in the context of the artist's history.
        
        Look for DISCREPANCIES between the expected style and the findings.
        
        Example:
        - If findings say "16kHz cutoff" (Low quality) and context says "Lo-fi Hip Hop from 1995", this is EXPECTED/NORMAL.
        - If findings say "Perfect Pitch" and context says "Raw Punk Band", this is SUSPICIOUS.
        
        Provide a "Critical Verification" summary."""
        
        # Serialize essential results for prompt
        metrics_summary = self._summarize_metrics(analysis_results)
        
        prompt = f"""
        CONTEXT:
        {context}
        
        TECHNICAL FINDINGS:
        {metrics_summary}
        
        Please provide your critical assessment. Is this likely AI-generated or just consistent with the genre/production?
        """
        
        return self.llm.generate(system, prompt)
        
    def _summarize_metrics(self, results: Dict) -> str:
        # Helper to format JSON into readable text for LLM
        summary = []
        
        # Assume results structure from analyzer.py
        if 'ai_probability' in results:
            summary.append(f"AI Probability: {results['ai_probability']:.2f}")
            
        if 'flags' in results:
            summary.append("Flags: " + ", ".join(results['flags']))
            
        # Add feature details if available
        return "\n".join(summary)


class PublicReporterAgent:
    """
    Writes the final "Human Friendly" report.
    """
    
    def __init__(self, llm_client: 'LLMClient'):
        self.llm = llm_client
        
    def write_report(self, technical_data: str, critique: str, context: str) -> str:
        """Generate the final HTML/Markdown report content."""
        if not self.llm.check_availability():
            return "LLM not available for report generation."
            
        system = """You are a Science Communicator for a general audience.
        Write a clear, engaging report about the authenticity of a music track.
        
        Structure:
        1. **Executive Summary**: Is it AI or Human? (Clear verdict).
        2. **The Artist Context**: Brief background.
        3. **Evidence**: Explain the technical findings in simple terms (e.g., explain what "spectral cutoff" means).
        4. **Conclusion**: Final thoughts.
        
        Use Markdown formatting."""
        
        prompt = f"""
        Context: {context}
        Critique: {critique}
        Technical Data: {technical_data}
        
        Write the public report.
        """
        
        return self.llm.generate(system, prompt, temperature=0.5)
