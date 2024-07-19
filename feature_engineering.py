import pandas as pd
import spotipy

def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 50):
        audio_features.extend(sp.audio_features(tracks=track_ids[i:i+50]))
    return audio_features

def prepare_data(df, audio_features):
    df_audio_features = pd.DataFrame(audio_features)
    numeric_features = df_audio_features.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
    df = df.join(numeric_features.set_index(df_audio_features['id']), on='track_id')

    # Ensure no NaNs in features
    df = df.dropna(subset=numeric_features.columns)

    # Check for NaNs and drop rows with NaNs in the target variable
    df = df.dropna(subset=['genres'])
    X = df[numeric_features.columns]
    y = df['genres']

    # Convert target variable to a categorical type
    y = y.astype('category').cat.codes

    return X, y, numeric_features
