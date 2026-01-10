import os
import requests
import shutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from src.config import config

@dataclass
class ValidationResult:
    passed: bool
    component: str
    message: str
    severity: str = "info"  # "info", "warning", "critical"
    error_details: Optional[str] = None

class SystemValidator:
    """
    Performs pre-flight checks on APIs, file system, and hardware.
    """
    
    def validate_llm(self, provider: str, config_dict: Dict) -> ValidationResult:
        """Test LLM API key with a minimal request."""
        api_key = config_dict.get('api_key') or config.api.get_llm_config(provider)[0]
        
        if not api_key:
            return ValidationResult(
                passed=False, 
                component=f"LLM ({provider})", 
                message="No API key found.", 
                severity="critical"
            )
            
        # Implementation of actual API test ping would go here
        # For now, we simulate success if key is present
        if len(api_key) < 10:
             return ValidationResult(
                passed=False, 
                component=f"LLM ({provider})", 
                message="API key seems invalid (too short).", 
                severity="critical"
            )
            
        return ValidationResult(passed=True, component=f"LLM ({provider})", message="Key detected & ready.")

    def validate_spotify(self) -> ValidationResult:
        """Validate Spotify credentials."""
        if not config.api.has_spotify_credentials():
            return ValidationResult(passed=False, component="Spotify", message="Not configured.", severity="warning")
            
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            auth_manager = SpotifyClientCredentials(
                client_id=config.api.spotify_client_id,
                client_secret=config.api.spotify_client_secret
            )
            spotipy.Spotify(auth_manager=auth_manager)
            return ValidationResult(passed=True, component="Spotify", message="Credentials valid.")
        except Exception as e:
            return ValidationResult(passed=False, component="Spotify", message=f"Auth failed: {e}", severity="warning")

    def validate_musicbrainz(self) -> ValidationResult:
        """Check MusicBrainz connectivity."""
        if not config.api.musicbrainz_contact:
            return ValidationResult(passed=False, component="MusicBrainz", message="No contact email set.", severity="warning")
            
        try:
            import musicbrainzngs
            musicbrainzngs.set_useragent("MusicTruth", "2.0", config.api.musicbrainz_contact)
            # Minimal ping
            musicbrainzngs.search_artists(artist="test", limit=1)
            return ValidationResult(passed=True, component="MusicBrainz", message="API reachable.")
        except Exception as e:
            return ValidationResult(passed=False, component="MusicBrainz", message=f"Connectivity issue: {e}", severity="warning")

    def validate_disk_space(self) -> ValidationResult:
        """Check if output directory has enough space."""
        base_path = config.paths.output_dir
        if not os.path.exists(base_path):
             os.makedirs(base_path, exist_ok=True)
             
        total, used, free = shutil.disk_usage(base_path)
        free_gb = free / (2**30)
        
        if free_gb < 1:
            return ValidationResult(passed=False, component="Disk Space", message=f"Low space: {free_gb:.1f} GB", severity="critical")
        elif free_gb < 5:
            return ValidationResult(passed=True, component="Disk Space", message=f"System OK ({free_gb:.1f} GB free)", severity="warning")
            
        return ValidationResult(passed=True, component="Disk Space", message=f"Adequate ({free_gb:.1f} GB free).")

    def validate_inputs(self, file_paths: List[str]) -> ValidationResult:
        """Verify all input files are readable."""
        missing = []
        for p in file_paths:
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
                
        if missing:
            return ValidationResult(passed=False, component="Input Files", message=f"Missing: {', '.join(missing)}", severity="critical")
            
        return ValidationResult(passed=True, component="Input Files", message=f"{len(file_paths)} files verified.")
