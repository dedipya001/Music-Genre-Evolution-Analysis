import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

scope = 'user-library-read'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

def get_track_data(sp, limit=50):
    results = sp.current_user_saved_tracks(limit=limit)
    tracks = results['items']
    track_data = []

    for item in tracks:
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
            'genres': genres
        })

    return track_data

track_data = get_track_data(sp)

with open('track_data.json', 'w') as f:
    json.dump(track_data, f)

with open('track_data.json', 'r') as f:
    track_data = json.load(f)

df = pd.DataFrame(track_data)

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df = df.explode('genres')

print(df.head())

print(df.describe())

genre_counts = df['genres'].value_counts()
print(genre_counts)

plt.figure(figsize=(14, 7))
genre_counts.plot(kind='bar')
plt.title('Distribution of Genres')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.show()

genre_trends = df.groupby([df['release_date'].dt.year, 'genres']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 7))
genre_trends.plot(kind='line', ax=plt.gca())
plt.title('Genre Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Tracks')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 7))
sns.heatmap(genre_trends.T, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Genre Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Genres')
plt.show()
