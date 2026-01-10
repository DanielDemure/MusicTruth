"""
History and Project Manager for MusicTruth.

Manages test sessions, project grouping, and history tracking.
Enables creating structured output directories and revisiting past results.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil

from ...config import config

def sanitize_name(name: str) -> str:
    """Sanitize string for use as directory name."""
    if not name:
        return "Unknown"
    # Replace common invalid chars
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, '')
    return name.strip().replace(" ", "_")

class HistoryManager:
    """
    Manages project history and results storage.
    
    Structure:
    output/
      ├── Project_Name/
      │   ├── 2024-05-20_14-30-00/
      │   │   ├── metadata.json
      │   │   ├── results.json
      │   │   ├── report.html
      │   │   └── ...
    """
    
    def __init__(self, output_root: Optional[str] = None):
        self.output_root = Path(output_root or config.paths.output_dir)
        self.current_project = "Default_Project"
        self.current_session_id = None
        
    def create_session(self, project_name: str = "Default_Project", 
                       artist: Optional[str] = None, 
                       album: Optional[str] = None) -> str:
        """
        Create a new analysis session.
        
        Args:
            project_name: Name of the project (folder)
            artist: Optional artist name for subfolder
            album: Optional album name for sub-subfolder
            
        Returns:
            Path to the session directory
        """
        self.current_project = sanitize_name(project_name)
        
        # Create timestamp string
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session_id = timestamp
        
        # Build path
        if artist and album:
            session_dir = self.output_root / sanitize_name(artist) / sanitize_name(album) / timestamp
        elif artist:
            session_dir = self.output_root / sanitize_name(artist) / timestamp
        else:
            session_dir = self.output_root / self.current_project / timestamp
            
        os.makedirs(session_dir, exist_ok=True)
        self.session_path = session_dir # Cache for get_session_dir
        
        return str(session_dir)
    
    def get_session_dir(self) -> str:
        """Get current session directory."""
        if hasattr(self, 'session_path'):
            return str(self.session_path)
            
        if not self.current_session_id:
            return self.create_session()
            
        return str(self.output_root / self.current_project / self.current_session_id)
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save analysis results to current session."""
        session_dir = self.get_session_dir()
        
        file_path = os.path.join(session_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=str)
            
        return file_path
        
    def list_projects(self) -> List[str]:
        """List all available projects."""
        if not self.output_root.exists():
            return []
            
        return [d.name for d in self.output_root.iterdir() if d.is_dir()]
    
    def list_sessions(self, project_name: str) -> List[str]:
        """List sessions for a project."""
        project_dir = self.output_root / project_name
        if not project_dir.exists():
            return []
            
        return [d.name for d in project_dir.iterdir() if d.is_dir()]
    
    def load_result(self, project_name: str, session_id: str, 
                   filename: str = "results.json") -> Optional[Dict[str, Any]]:
        """Load specific result file."""
        file_path = self.output_root / project_name / session_id / filename
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading result: {e}")
            return None
            
    def export_project(self, project_name: str, export_path: str):
        """Export entire project as zip."""
        project_dir = self.output_root / project_name
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory {project_name} not found")
            
        shutil.make_archive(export_path, 'zip', project_dir)

# Global instance
history_manager = HistoryManager()
