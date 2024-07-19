from data_retrival import get_all_track_data, save_track_data, load_track_data
from feature_engineering import get_audio_features, prepare_data
from visualization import plot_genre_distribution, plot_genre_trends, plot_genre_heatmap, plot_correlation_matrix
from modelling import train_and_evaluate_model, evaluate_model_with_loaded_model

import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import joblib

def main():
    load_dotenv()
    
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

    scope = 'user-library-read'
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope=scope))
    track_data = get_all_track_data(sp)
    save_track_data(track_data)
    track_data = load_track_data()
    
    # Create DataFrame
    df = pd.DataFrame(track_data)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.explode('genres')

    # Get audio features
    track_ids = df['track_id'].tolist()
    audio_features = get_audio_features(sp, track_ids)
    X, y, numeric_features = prepare_data(df, audio_features)

    # Visualizations
    plot_genre_distribution(df)
    plot_genre_trends(df)
    genre_trends = df.groupby([df['release_date'].dt.year, 'genres']).size().unstack(fill_value=0)
    plot_genre_heatmap(genre_trends)
    plot_correlation_matrix(numeric_features)

    # Train model and save it
    train_and_evaluate_model(X, y)
    
    # Load model and evaluate
    evaluate_model_with_loaded_model(X, y)

if __name__ == "__main__":
    main()
