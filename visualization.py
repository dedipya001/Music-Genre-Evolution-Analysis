import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_genre_distribution(df):
    genre_counts = df['genres'].value_counts()
    plt.figure(figsize=(14, 7))
    genre_counts.plot(kind='bar')
    plt.title('Distribution of Genres')
    plt.xlabel('Genres')
    plt.ylabel('Count')
    plt.show()

def plot_genre_trends(df):
    genre_trends = df.groupby([df['release_date'].dt.year, 'genres']).size().unstack(fill_value=0)
    plt.figure(figsize=(14, 7))
    genre_trends.plot(kind='line', ax=plt.gca())
    plt.title('Genre Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Tracks')
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_genre_heatmap(genre_trends):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    sns.heatmap(genre_trends.T, cmap='coolwarm', cbar=True)
    plt.title('Heatmap of Genre Popularity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Genres')
    plt.show()

def plot_correlation_matrix(numeric_features):
    correlation_matrix = numeric_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Audio Features')
    plt.show()
