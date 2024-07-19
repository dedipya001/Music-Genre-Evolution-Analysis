import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# def visualize_classification_report(y_true, y_pred):
#     report = classification_report(y_true, y_pred, output_dict=True)
#     report_df = pd.DataFrame(report).transpose()

#     plt.figure(figsize=(12, 8))
#     sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title('Classification Report')
#     plt.xlabel('Metrics')
#     plt.ylabel('Classes')
#     plt.show()

def plot_correlation_matrix(numeric_features):
    correlation_matrix = numeric_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Audio Features')
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    y_true = np.array(y_true).astype(str)
    y_pred = np.array(y_pred).astype(str)
    
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # Ensure class_names are valid labels
    class_names = [str(name) for name in class_names if str(name) in unique_labels]
    
    if not class_names:
        print(f"Unique labels in y_true and y_pred: {unique_labels}")
        print(f"Provided class_names: {class_names}")
        raise ValueError("The 'class_names' parameter does not contain any labels present in 'y_true' and 'y_pred'")

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()



def summarize_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    summary = {
        "Accuracy": report["accuracy"],
        "Macro Average": report["macro avg"],
        "Micro Average": report["micro avg"],
        "Weighted Average": report["weighted avg"]
    }
    return summary




def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()



def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print(f"AUC: {roc_auc_score(y_true, y_scores)}")
