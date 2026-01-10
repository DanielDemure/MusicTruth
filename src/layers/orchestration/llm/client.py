import os
import json
import requests
from typing import Optional, Dict, Any, List

# Try importing SDKs
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class LLMClient:
    """
    Unified client for LLM interactions across multiple providers.
    Supports: OpenAI, Anthropic, Google Gemini, DeepSeek, Ollama, LM Studio.
    """
    
    def __init__(self, provider: str = "openai", model: str = None, 
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        
        # Set default models if not provided
        if not model:
            self.model = self._get_default_model(self.provider)
        else:
            self.model = model
            
        self._init_client()
        
    def _get_default_model(self, provider: str) -> str:
        defaults = {
            "openai": "gpt-4-turbo",
            "anthropic": "claude-3-opus-20240229",
            "gemini": "gemini-1.5-pro",
            "deepseek": "deepseek-chat",
            "ollama": "llama3",
            "lm_studio": "local-model",
        }
        return defaults.get(provider, "gpt-3.5-turbo")

    def _init_client(self):
        """Initialize the specific provider client."""
        if self.provider == "openai":
            if OPENAI_AVAILABLE:
                self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
            else:
                print("Warning: openai library not installed.")

        elif self.provider == "anthropic":
            if ANTHROPIC_AVAILABLE:
                self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if self.api_key:
                    self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                print("Warning: anthropic library not installed.")
                
        elif self.provider == "gemini":
            if GEMINI_AVAILABLE:
                self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model)
            else:
                print("Warning: google-generativeai library not installed.")
        
        elif self.provider == "deepseek":
            if OPENAI_AVAILABLE:
                self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
                self.base_url = self.base_url or "https://api.deepseek.com/v1"
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                print("Warning: openai library needed for DeepSeek.")

        elif self.provider == "openrouter":
            if OPENAI_AVAILABLE:
                self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
                self.base_url = self.base_url or "https://openrouter.ai/api/v1"
                if self.api_key:
                    self.client = OpenAI(
                        api_key=self.api_key, 
                        base_url=self.base_url,
                        default_headers={"HTTP-Referer": "https://musictruth.ai", "X-Title": "MusicTruth"}
                    )
            else:
                print("Warning: openai library needed for OpenRouter.")

        elif self.provider == "custom":
            if OPENAI_AVAILABLE:
                # User must provide base_url and key
                if self.base_url and self.api_key:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                print("Warning: openai library needed for Custom provider.")

        elif self.provider in ["ollama", "lm_studio", "local"]:
            # Local providers often use OpenAI-compatible endpoints
            if OPENAI_AVAILABLE:
                if not self.base_url:
                    if self.provider == "ollama":
                        self.base_url = "http://localhost:11434/v1"
                    else:
                        self.base_url = "http://localhost:1234/v1"
                
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="lm-studio"  # Often ignored but required
                )
            else:
                pass

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Generate text from LLM."""
        if not self.check_availability():
            return f"LLM Client not initialized for {self.provider}."
            
        try:
            if self.provider in ["openai", "deepseek", "ollama", "lm_studio", "local", "openrouter", "custom"]:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.content[0].text
                
            elif self.provider == "gemini":
                # Gemeni system prompts are often set at model config or just prepended
                # We'll prepend for simplicity in this wrapper
                full_prompt = f"System Instruction: {system_prompt}\n\nUser Request: {user_prompt}"
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    )
                )
                return response.text
                
        except Exception as e:
            return f"LLM Generation Error ({self.provider}): {str(e)}"

    def check_availability(self) -> bool:
        return self.client is not None or (self.provider == "gemini" and GEMINI_AVAILABLE and self.api_key)
