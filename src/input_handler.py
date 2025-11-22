import os
import subprocess
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.aiff'}

def scan_files(directory: str) -> List[str]:
    """
    Recursively scans the directory for supported audio files.
    """
    audio_files = []
    path = Path(directory)
    
    if not path.exists():
        print(f"Warning: Directory {directory} does not exist.")
        return []

    for file_path in path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(str(file_path.absolute()))
            
    return audio_files

def download_from_sources(sources_file: str, output_dir: str) -> List[str]:
    """
    Reads URLs from sources_file and downloads them using yt-dlp.
    Returns a list of downloaded file paths.
    """
    downloaded_files = []
    
    if not os.path.exists(sources_file):
        print(f"Warning: Sources file {sources_file} not found.")
        return []

    with open(sources_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not urls:
        return []

    print(f"Found {len(urls)} URLs to process...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        print(f"Processing URL: {url}")
        try:
            # Construct yt-dlp command
            # We use a specific template to avoid overwriting and to get clean filenames
            # -x: Extract audio
            # --audio-format mp3: Convert to mp3 (for compatibility, though flac is better for analysis, 
            # but youtube audio is usually lossy anyway. Let's stick to best quality available)
            # Actually, for analysis, we want the best quality. 
            # --audio-quality 0: Best quality
            
            cmd = [
                'yt-dlp',
                '-x',
                '--audio-format', 'mp3', # Keeping it simple for MVP, maybe flac later if needed
                '--audio-quality', '0',
                '-o', f'{output_dir}/%(title)s.%(ext)s',
                '--no-playlist',
                url
            ]
            
            subprocess.run(cmd, check=True)
            
            # We need to find what was downloaded. 
            # This is a bit tricky with yt-dlp output templates.
            # For now, we will rescan the output directory after download.
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {url}: {e}")
        except FileNotFoundError:
            print("Error: yt-dlp not found. Please install yt-dlp and add it to your PATH.")
            break

    # Rescan the output directory to get the new files
    # This is a simple way to return the paths
    return scan_files(output_dir)

if __name__ == "__main__":
    # Test
    print("Scanning input/...")
    files = scan_files("input")
    print(f"Found {len(files)} files.")
