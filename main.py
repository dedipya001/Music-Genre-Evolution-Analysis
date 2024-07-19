from data_retrival import get_all_track_data, save_track_data, load_track_data
from feature_engineering import get_audio_features, prepare_data
from modelling import train_and_evaluate_model, evaluate_model_with_loaded_model, predict_genre, recommend_tracks, load_model
from visualization import plot_genre_distribution, plot_genre_trends, plot_genre_heatmap, plot_correlation_matrix,plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

# load_dotenv()


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
    
    df = pd.DataFrame(track_data)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.explode('genres')

    track_ids = df['track_id'].tolist()
    audio_features = get_audio_features(sp, track_ids)
    X, y, numeric_features = prepare_data(df, audio_features)

    plot_genre_distribution(df)
    plot_genre_trends(df)
    genre_trends = df.groupby([df['release_date'].dt.year, 'genres']).size().unstack(fill_value=0)
    plot_genre_heatmap(genre_trends)
    plot_correlation_matrix(numeric_features)

    model = load_model('random_forest_model.pkl')
    
    genre_to_recommend = 'Rock'  
    recommended_tracks = recommend_tracks(df, genre_to_recommend)
    print(f"Recommended Tracks for {genre_to_recommend}:\n{recommended_tracks}")

    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # y_pred = model.predict(X_test)

    # class_names = list(df['genres'].unique())
    # class_names = [name for name in class_names if name in y_test]  # Ensure class_names are present in y_test
    
    # plot_confusion_matrix(y_test, y_pred, class_names)
    
    # summary = summarize_classification_report(y_test, y_pred)
    # print(f"Classification Report Summary: {summary}")

    # plot_precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    # plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    



if __name__ == "__main__":
    main()