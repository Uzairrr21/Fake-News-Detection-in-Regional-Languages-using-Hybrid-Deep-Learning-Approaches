import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set visualization parameters for research paper quality
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Standard font for plot text

# ==============================================================================
# PART 1: Data Preparation and Distribution Visualization Functions
# ==============================================================================

FILE_NAME = '/kaggle/input/datasets/uzairmoazzam203/research-paper/ml_ready_2column.csv'

def visualize_data_distribution(df):
    """Visualizes class distribution and text length, fixing the FutureWarning."""

    # --- A. Class Distribution ---
    plt.figure(figsize=(8, 5))
    # FIX: Assigned 'label' to 'hue' and set legend=False to resolve FutureWarning
    ax = sns.countplot(y='label', data=df, order=df['label'].value_counts().index, palette="viridis", hue='label', legend=False)

    # Add counts on the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_width()}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=10)

    plt.title('Distribution of News Categories (Classes)', fontsize=14)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    print("Saved: class_distribution.png")

    # --- B. Text Length Distribution ---
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', hue='label', element='step', kde=True, palette="rocket")

    # Calculate median lengths for annotation
    medians = df.groupby('label')['text_length'].median().to_dict()

    plt.title('Word Count Distribution by News Category', fontsize=14)
    plt.xlabel('Word Count (Text Length)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Label', labels=[f'{k} (Median: {v:.0f} words)' for k, v in medians.items()])
    plt.tight_layout()
    plt.savefig('text_length_distribution.png')
    plt.close()
    print("Saved: text_length_distribution.png")


# ==============================================================================
# PART 2: Model Performance Visualization Functions
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Transformer Model"):
    """Plots a Confusion Matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                cbar=False, annot_kws={"size": 14})

    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_classification_metrics(report_df, class_names, model_name="Transformer Model"):
    """Plots Precision, Recall, and F1-Score for each class."""
    metrics = ['precision', 'recall', 'f1-score']

    plt.figure(figsize=(10, 6))

    # Filter for only the class names
    plot_df = report_df.loc[class_names, metrics].reset_index().rename(columns={'index': 'Class'})
    plot_df_melted = plot_df.melt(id_vars='Class', var_name='Metric', value_name='Score')

    ax = sns.barplot(x='Class', y='Score', hue='Metric', data=plot_df_melted, palette="magma")

    # Add score values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=9)

    plt.ylim(0, 1.1)
    plt.title(f'Classification Metrics by Class for {model_name}', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('classification_metrics_bar_chart.png')
    plt.close()
    print("Saved: classification_metrics_bar_chart.png")


# ==============================================================================
# PART 3: Simulation Function
# ==============================================================================

def simulate_article_classification(new_article_text, vectorizer, model):
    """
    Simulates the classification of a new, unseen article.
    """
    print(f"\n--- New Article Simulation ---")
    print(f"Input Article: '{new_article_text}'")

    # Vectorize the new text
    new_article_vec = vectorizer.transform([new_article_text])

    # Predict the label and probability
    prediction = model.predict(new_article_vec)[0]
    proba = model.predict_proba(new_article_vec)[0]

    # Map probabilities to class names
    proba_map = {str(class_name): f"{proba[i]*100:.2f}%" for i, class_name in enumerate(model.classes_)}
    sorted_proba = dict(sorted(proba_map.items(), key=lambda item: item[1], reverse=True))

    # Output result
    print(f"Predicted Label: {prediction}")
    print(f"Confidence Scores: {sorted_proba}")
    print("----------------------------")
    return prediction


# ==============================================================================
# EXECUTION (Baseline Model Training and Visualization)
# ==============================================================================

if __name__ == '__main__':
    try:
        df = pd.read_csv(FILE_NAME)
        # Ensure all columns are strings (critical for text data)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(str)
        print(f"Successfully loaded '{FILE_NAME}'.")

    except FileNotFoundError:
        print(f"Error: The file '{FILE_NAME}' was not found.")
        exit()

    # Determine unique class names for consistency
    CLASS_NAMES = sorted(df['label'].unique().tolist())

    # 1. Data Distribution Plots
    print("\n--- Generating Data Distribution Plots ---")
    visualize_data_distribution(df.copy()) # Use a copy to avoid SettingWithCopyWarning

    # 2. Data Splitting and Feature Engineering
    X = df['text']
    y = df['label']
    # Stratified split ensures equal class representation in test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Model Training (Baseline: Multinomial Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # 4. Detailed Metrics Output
    print("\n--- Detailed Model Performance Metrics ---")

    # Classification Report
    print("\nClassification Report (Test Data):")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix Counts
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("\nConfusion Matrix Counts:")
    print(cm_df)

    report_df = pd.DataFrame(report).transpose()

    # 5. Model Performance Plots
    print("\n--- Generating Model Performance Plots ---")
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, model_name="Baseline Naive Bayes Model")
    plot_classification_metrics(report_df, CLASS_NAMES, model_name="Baseline Naive Bayes Model")

    # 6. Run Simulations
    new_article_A = "پاکستان کے وزیر خارجہ نے اقوام متحدہ میں کشمیر پر اہم قرارداد پیش کر دی ہے۔"
    simulate_article_classification(new_article_A, vectorizer, model)

    new_article_B = "کرکٹ ٹیم کے کپتان نے اعلان کیا ہے کہ وہ اب صرف سونا اور ہیرے کھائیں گے۔"
    simulate_article_classification(new_article_B, vectorizer, model)

    print("\nScript execution complete. Check the current directory for the generated PNG files.")