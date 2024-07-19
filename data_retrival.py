import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Spotify API credentials
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

# Define scope
scope = 'user-library-read'

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

def get_all_track_data(sp, limit=50):
    track_data = []
    offset = 0
    while True:
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)
        if not results['items']:
            break
        for item in results['items']:
            track = item['track']
            artist_id = track['artists'][0]['id']
            artist_name = track['artists'][0]['name']
            track_name = track['name']
            album_name = track['album']['name']
            release_date = track['album']['release_date']
            artist_info = sp.artist(artist_id)
            genres = artist_info['genres']
            track_data.append({
                'track_name': track_name,
                'artist_name': artist_name,
                'album_name': album_name,
                'release_date': release_date,
                'genres': genres,
                'track_id': track['id']
            })
        offset += limit
    return track_data

def save_track_data(track_data, filename='track_data.json'):
    with open(filename, 'w') as f:
        json.dump(track_data, f)

def load_track_data(filename='track_data.json'):
    with open(filename, 'r') as f:
        return json.load(f)
