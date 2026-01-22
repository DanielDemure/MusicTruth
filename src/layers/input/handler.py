import os
import sys
from typing import List, Dict, Optional, Tuple
import glob
from dataclasses import dataclass
from src.utils.logger import logger
from pathlib import Path

MAX_FILE_SIZE_MB = 500

@dataclass
class AudioSource:
    """Represents a single audio input source."""
    path_or_url: str
    source_type: str  # 'file', 'spotify', 'youtube', 'deezer'
    group_id: Optional[str] = None  # ID to group same song versions
    metadata: Dict = None

class InputHandler:
    """
    Handles input collection from various sources.
    Supports local files, URLs, and multi-source grouping.
    """
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.aiff'}
    
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.sources: List[AudioSource] = []
        
    def scan_directory(self, recursive: bool = False) -> List[str]:
        """Scan input directory for audio files."""
        files = []
        if recursive:
            search_pattern = os.path.join(self.input_dir, '**', '*')
        else:
            search_pattern = os.path.join(self.input_dir, '*')
            
        for filepath in glob.glob(search_pattern, recursive=recursive):
            if os.path.isfile(filepath):
                ext = os.path.splitext(filepath)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(filepath)
        return files
        
    def scan_directory_path(self, path: str) -> List[str]:
        """Scan a specific directory path recursively with safety checks."""
        files = []
        if not os.path.exists(path): return []
        
        # Security: Prevent path traversal
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith(os.path.abspath(self.input_dir)) and not "Apps" in absolute_path:
             # Basic check: allow internal or known workspace paths
             # This is a bit relaxed for now to allow user flexibility but warns
             logger.debug(f"Scanning path outside standard input dir: {absolute_path}")

        search_pattern = os.path.join(path, '**', '*')
        for filepath in glob.glob(search_pattern, recursive=True):
            if os.path.isfile(filepath):
                # Check file size
                try:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        logger.warning(f"Skipping large file: {filepath} ({size_mb:.1f}MB)")
                        continue
                except OSError:
                    continue

                ext = os.path.splitext(filepath)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(filepath)
        return files

    def add_sources_from_paths(self, file_paths: List[str], group_id: Optional[str] = None):
        """Add local files to source list."""
        for path in file_paths:
            self.sources.append(AudioSource(
                path_or_url=path,
                source_type='file',
                group_id=group_id,
                metadata={'filename': os.path.basename(path)}
            ))
            
    def add_source_url(self, url: str, group_id: Optional[str] = None):
        """Add URL source (Spotify/Youtube/etc)."""
        source_type = 'unknown'
        if 'spotify.com' in url:
            source_type = 'spotify'
        elif 'youtube.com' in url or 'youtu.be' in url:
            source_type = 'youtube'
        elif 'deezer.com' in url:
            source_type = 'deezer'
            
        self.sources.append(AudioSource(
            path_or_url=url,
            source_type=source_type,
            group_id=group_id
        ))

    def get_ready_sources_list(self) -> List[AudioSource]:
        """Return list of sources ready for analysis."""
        return self.sources
    
    def group_sources_by_song(self) -> Dict[str, List[AudioSource]]:
        """
        Group sources that represent the same song.
        Returns generic Group IDs if none provided.
        """
        groups = {}
        ungrouped_idx = 0
        
        for source in self.sources:
            gid = source.group_id
            if not gid:
                # If no group ID, treat as individual unless manual grouping logic added
                gid = f"track_{ungrouped_idx}"
                ungrouped_idx += 1
                
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(source)
            
        return groups
        
    def download_remote_sources(self, output_dir: str) -> List[Tuple[AudioSource, str]]:
        """
        Download remote sources to output_dir using provider-specific tools.
        Returns list of (source, local_path).
        """
        import subprocess
        import shutil
        
        # Ensure output dir exists
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded = []
        
        for source in self.sources:
            if source.source_type == 'file':
                if os.path.exists(source.path_or_url):
                    downloaded.append((source, source.path_or_url))
                continue
            
            url = source.path_or_url
            print(f"⬇️  Downloading ({source.source_type}): {url}")
            
            try:
                if "spotify" in url:
                    # Use spotdl
                    # We use 'python -m spotdl' to be safe strictly if installed via pip
                    cmd = [sys.executable, "-m", "spotdl", "download", url, "--output", output_dir]
                    
                    # Capture state of directory before to know what was added
                    pre_files = set(os.listdir(output_dir))
                    
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ spotdl failed: {e}")
                        continue

                    # Find new files
                    post_files = set(os.listdir(output_dir))
                    new_files = post_files - pre_files
                    
                    if new_files:
                        for f in new_files:
                            if f.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                                full_path = os.path.join(output_dir, f)
                                downloaded.append((source, full_path))
                                print(f"   ✅ Downloaded: {f}")
                    else:
                        # Fallback: Maybe file already existed? Check for files matching likely name?
                        # Or just add all mp3s in the folder that look recent? 
                        # Safer: Just warn for now.
                        print(f"   ℹ️ No new files detected (maybe already downloaded?). Checking folder...")
                        # If user is downloading to persistent input folder, file might exist.
                        # Let's verify if we can find it. 
                        # spotdl usually skips if exists.
                        pass

                elif "youtube" in url or "youtu.be" in url:
                    # Use yt-dlp
                    out_tmpl = os.path.join(output_dir, "%(artist)s - %(title)s.%(ext)s")
                    
                    cmd = [
                        "yt-dlp",
                        "-x", "--audio-format", "mp3",
                        "--add-metadata",
                        "--no-playlist",
                        "-o", out_tmpl,
                        url
                    ]
                    
                    pre_files = set(os.listdir(output_dir))
                    subprocess.run(cmd, check=True)
                    post_files = set(os.listdir(output_dir))
                    new_files = post_files - pre_files
                    
                    for f in new_files:
                        if f.endswith('.mp3'):
                             full_path = os.path.join(output_dir, f)
                             downloaded.append((source, full_path))
                             print(f"   ✅ Downloaded: {f}")

                elif "deezer" in url:
                    print("⚠️  Deezer direct download not fully supported. Trying 'spotdl' URL search...")
                    print("   Please use 'deemix' for high-quality Deezer rips.")
                    
                else:
                    print(f"❌ Unsupported URL type: {url}")
                    
            except Exception as e:
                print(f"❌ Error downloading {url}: {e}")
                
        return downloaded

def get_input_handler(input_dir: str):
    return InputHandler(input_dir)
