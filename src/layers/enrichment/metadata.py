"""
Metadata enrichment from external APIs.

Integrates Spotify, MusicBrainz, and other music databases.
"""

from typing import Optional, Dict, Any
import os

# Spotify
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False

# MusicBrainz
try:
    import musicbrainzngs as mb
    MUSICBRAINZ_AVAILABLE = True
except ImportError:
    MUSICBRAINZ_AVAILABLE = False


class MetadataEnricher:
    """
    Fetches metadata from multiple sources to enrich analysis context.
    """
    
    def __init__(self):
        self.spotify_client = None
        self.mb_configured = False
        
        # Initialize Spotify
        if SPOTIFY_AVAILABLE:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if client_id and client_secret:
                try:
                    auth_manager = SpotifyClientCredentials(
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
                except Exception as e:
                    print(f"⚠️ Spotify init failed: {e}")
        
        # Initialize MusicBrainz
        if MUSICBRAINZ_AVAILABLE:
            mb.set_useragent("MusicTruth", "2.0", "https://github.com/DanielDemure/MusicTruth")
            self.mb_configured = True
    
    def enrich_from_spotify(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata from Spotify.
        
        Returns:
            Dict with keys: popularity, genres, release_date, etc.
        """
        if not self.spotify_client:
            return None
        
        try:
            # Search for track
            query = f"artist:{artist} track:{title}"
            results = self.spotify_client.search(q=query, type='track', limit=1)
            
            if not results['tracks']['items']:
                return None
            
            track = results['tracks']['items'][0]
            
            # Get artist details
            artist_id = track['artists'][0]['id']
            artist_info = self.spotify_client.artist(artist_id)
            
            return {
                'spotify_id': track['id'],
                'popularity': track['popularity'],
                'release_date': track['album']['release_date'],
                'genres': artist_info.get('genres', []),
                'artist_popularity': artist_info.get('popularity'),
                'followers': artist_info.get('followers', {}).get('total'),
                'preview_url': track.get('preview_url'),
                'explicit': track.get('explicit', False)
            }
            
        except Exception as e:
            print(f"⚠️ Spotify lookup failed: {e}")
            return None
    
    def enrich_from_musicbrainz(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata from MusicBrainz.
        
        Returns:
            Dict with keys: mbid, country, label, etc.
        """
        if not self.mb_configured:
            return None
        
        try:
            # Search for recording
            results = mb.search_recordings(
                artist=artist,
                recording=title,
                limit=1
            )
            
            if not results['recording-list']:
                return None
            
            recording = results['recording-list'][0]
            
            return {
                'mbid': recording.get('id'),
                'title': recording.get('title'),
                'artist_credit': recording.get('artist-credit-phrase'),
                'length_ms': recording.get('length'),
                'score': recording.get('ext:score')  # Match confidence
            }
            
        except Exception as e:
            print(f"⚠️ MusicBrainz lookup failed: {e}")
            return None
    
    def enrich(self, artist: str, title: str) -> Dict[str, Any]:
        """
        Fetch metadata from all available sources.
        
        Returns:
            Combined metadata dict
        """
        metadata = {
            'artist': artist,
            'title': title,
            'spotify': None,
            'musicbrainz': None
        }
        
        # Try Spotify
        spotify_data = self.enrich_from_spotify(artist, title)
        if spotify_data:
            metadata['spotify'] = spotify_data
        
        # Try MusicBrainz
        mb_data = self.enrich_from_musicbrainz(artist, title)
        if mb_data:
            metadata['musicbrainz'] = mb_data
        
        return metadata


# Singleton instance
metadata_enricher = MetadataEnricher()
