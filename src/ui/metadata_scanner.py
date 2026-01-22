import os
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.mp4 import MP4
from typing import List, Dict, Optional
import questionary
from questionary import Choice
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.config import config
from src.utils.logger import logger

console = Console()

class MetadataScanner:
    """
    Handles metadata extraction from files and matching with online APIs.
    """
    
    def scan_files(self, file_paths: List[str]) -> List[Dict]:
        """
        Scan a list of file path strings for basic metadata.
        """
        results = []
        for path in file_paths:
            metadata = {
                'file_path': path,
                'filename': os.path.basename(path),
                'artist': 'Unknown Artist',
                'album': 'Unknown Album',
                'title': os.path.splitext(os.path.basename(path))[0],
                'source': 'filename'
            }
            
            try:
                audio = mutagen.File(path, easy=True)
                if audio:
                    if 'artist' in audio: metadata['artist'] = audio['artist'][0]
                    if 'album' in audio: metadata['album'] = audio['album'][0]
                    if 'title' in audio: metadata['title'] = audio['title'][0]
                    metadata['source'] = 'id3'
            except Exception as e:
                # Fallback to filename parsing
                logger.debug(f"ID3 extraction failed for {path}: {e}")
                
            results.append(metadata)
        return results

    def auto_fetch_metadata(self, tracks: List[Dict]) -> List[Dict]:
        """
        Attempt to enrich metadata using MusicBrainz and Spotify.
        """
        can_mb = bool(config.api.musicbrainz_contact)
        can_spot = config.api.has_spotify_credentials()
        
        if not (can_mb or can_spot):
            return tracks
            
        rprint(f"\n[bold blue]Auto-fetching metadata for {len(tracks)} tracks...[/bold blue]")
        
        # Initialize MusicBrainz
        if can_mb:
            import musicbrainzngs
            musicbrainzngs.set_useragent("MusicTruth", "2.0", config.api.musicbrainz_contact)
            
        # Initialize Spotify (simplified)
        sp = None
        if can_spot:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=config.api.spotify_client_id,
                    client_secret=config.api.spotify_client_secret
                )
                sp = spotipy.Spotify(auth_manager=auth_manager)
            except Exception as e:
                logger.warning(f"Spotify auth failed: {e}")
                rprint(f"[yellow]⚠️ Spotify auth failed: {e}[/yellow]")
                can_spot = False

        for track in tracks:
            # Improvement: If we have a folder name that looks like metadata, use it
            folder_hint = ""
            if 'file_path' in track:
                dir_name = os.path.basename(os.path.dirname(track['file_path']))
                if " - " in dir_name:
                    folder_hint = dir_name

            query = f"artist:{track['artist']} recording:{track['title']}"
            if folder_hint and track['artist'] == 'Unknown Artist':
                rprint(f"  [dim]Using folder hint: {folder_hint}[/dim]")
                query = f"{folder_hint} {track['title']}"

            rprint(f"  🔍 Searching: [dim]{track['artist']} - {track['title']}[/dim]")
            
            # 1. MusicBrainz Lookup
            if can_mb:
                try:
                    import musicbrainzngs
                    # Limit search but check scores
                    result = musicbrainzngs.search_recordings(query=query, limit=5)
                    if result['recording-list']:
                        # Verification: Check if it's a "real" match or just noise
                        # MusicBrainz returns 'ext:score'
                        top_match = result['recording-list'][0]
                        score = int(top_match.get('ext:score', '0'))
                        
                        if score > 80: # Reasonable threshold
                            track['artist'] = top_match['artist-credit'][0]['name']
                            track['title'] = top_match['title']
                            if 'release-list' in top_match:
                                track['album'] = top_match['release-list'][0]['title']
                            track['source'] = 'MusicBrainz'
                            continue 
                        else:
                            rprint(f"    [dim]Low confidence match ({score}). skipping MB.[/dim]")
                except Exception:
                    pass
            
            # 2. Spotify Lookup (Often more accurate for modern titles)
            if can_spot and sp:
                try:
                    q = f"track:{track['title']} artist:{track['artist']}"
                    if track['artist'] == 'Unknown Artist' and folder_hint:
                        q = f"{folder_hint} {track['title']}"
                        
                    res = sp.search(q=q, type='track', limit=1)
                    if res['tracks']['items']:
                        item = res['tracks']['items'][0]
                        track['artist'] = item['artists'][0]['name']
                        track['title'] = item['name']
                        track['album'] = item['album']['name']
                        track['source'] = 'Spotify'
                except Exception:
                    pass
                
        return tracks

    def interactive_review(self, tracks: List[Dict]) -> List[Dict]:
        """
        Provide an interactive wizard to review and correct metadata.
        """
        reviewed_tracks = []
        
        console.rule("[bold cyan]Metadata Review[/bold cyan]")
        rprint("[dim]Review and correct metadata for your sources.[/dim]\n")
        
        for i, track in enumerate(tracks):
            table = Table(title=f"Track {i+1}/{len(tracks)}: {track['filename']}")
            table.add_column("Field", style="bold")
            table.add_column("Value", style="green")
            table.add_column("Source", style="dim")
            
            table.add_row("Artist", track['artist'], track['source'])
            table.add_row("Album", track['album'], track['source'])
            table.add_row("Title", track['title'], track['source'])
            
            console.print(table)
            
            choices = [
                "Accept",
                "Edit Artist",
                "Edit Album",
                "Edit Title",
                Choice("Apply Artist/Album to ALL TRACKS", value="apply_all"),
                Choice("Apply Artist/Album to REMAINING", value="apply_remaining"),
                "Skip Track"
            ]
            
            while True:
                choice = questionary.select(
                    "Action:",
                    choices=choices,
                    default="Accept"
                ).ask()
                
                if choice == "Accept":
                    reviewed_tracks.append(track)
                    break
                elif choice == "apply_all":
                    # Apply to ALL (including already reviewed if we wanted, but let's stick to simple)
                    for t in tracks:
                        t['artist'] = track['artist']
                        t['album'] = track['album']
                        t['source'] = 'manual (global)'
                    rprint(f"[green]✅ Applied '{track['artist']} - {track['album']}' to all tracks.[/green]")
                elif choice == "apply_remaining":
                    for t in tracks[i:]:
                        t['artist'] = track['artist']
                        t['album'] = track['album']
                        t['source'] = 'manual (global)'
                    rprint(f"[green]✅ Applied '{track['artist']} - {track['album']}' to remaining tracks.[/green]")
                elif choice == "Edit Artist":
                    track['artist'] = questionary.text("Artist:", default=track['artist']).ask()
                    track['source'] = 'manual'
                elif choice == "Edit Album":
                    track['album'] = questionary.text("Album:", default=track['album']).ask()
                    track['source'] = 'manual'
                elif choice == "Edit Title":
                    track['title'] = questionary.text("Title:", default=track['title']).ask()
                    track['source'] = 'manual'
                elif choice == "Skip Track":
                    reviewed_tracks.append(track)
                    break

        return reviewed_tracks

        return reviewed_tracks
