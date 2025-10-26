# ============================================
# PHISHING EMAIL DETECTION PIPELINE (NO BERT)
# ============================================

# Step 0: Suppress warnings and configure environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Differential Privacy Library
from diffprivlib.models import LogisticRegression as DPLogisticRegression

# ============================================
# Step 2: Load and Clean Data
# ============================================

df = pd.read_csv('Cleaned_PhishingEmailData.csv', encoding='ISO-8859-1')
df['Email_Content'] = df['Email_Content'].fillna("").astype(str)
df['Email_Subject'] = df['Email_Subject'].fillna("").astype(str)
df['Label'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "link", text)
    text = re.sub(r"\S+@\S+", "email", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['clean_content'] = df['Email_Content'].apply(clean_text)

# ============================================
# Step 3: Feature Engineering
# ============================================

df['subject_length'] = df['Email_Subject'].apply(lambda x: len(str(x)))
df['link_count'] = df['Email_Content'].apply(lambda x: len(re.findall(r"http\S+", str(x))))
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0)

# ============================================
# Step 4: Exploratory Data Analysis (EDA)
# ============================================

sns.histplot(df['subject_length'], bins=30, kde=True)
plt.title("Distribution of Subject Lengths")
plt.xlabel("Subject Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

sns.countplot(x='has_link', data=df)
plt.title("Presence of Links in Emails")
plt.xlabel("Has Link")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================
# Step 5: Clustering (Feature Visualization)
# ============================================

features = ['subject_length', 'link_count', 'has_link']
X_cluster = df[features]

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

# Multiple clustering algorithms
clustering_algorithms = {
    'K-Means': KMeans(n_clusters=2, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=2),
    'MeanShift': MeanShift(),
    'Spectral': SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=42),
    'GMM': GaussianMixture(n_components=2, random_state=42)
}

for name, algorithm in clustering_algorithms.items():
    clusters = algorithm.fit_predict(X_cluster)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set2')
    plt.title(f"{name} Clustering with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

# ============================================
# Step 6: Train/Test Split and Scaling
# ============================================

X = df[features]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Step 7: Evaluation Function
# ============================================

def evaluate_model(y_true, y_pred):
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = None
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc
    }

def plot_metrics(name, metrics):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='Blues')
    plt.title(f"{name} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================
# Step 8: Classical Machine Learning Models
# ============================================

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    metrics = evaluate_model(y_test, y_pred)
    results[name] = metrics
    plot_metrics(name, metrics)

# ============================================
# Step 9: LSTM Model (Deep Learning)
# ============================================

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_content'])
X_seq = tokenizer.texts_to_sequences(df['clean_content'])
X_pad = pad_sequences(X_seq, maxlen=100)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_pad, y, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(Embedding(5000, 128))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=3, batch_size=64, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
predictions['LSTM'] = y_pred_lstm

print("\nLSTM Classification Report:\n")
print(classification_report(y_test_lstm, y_pred_lstm, zero_division=0))
metrics_lstm = evaluate_model(y_test_lstm, y_pred_lstm)
results['LSTM'] = metrics_lstm
plot_metrics("LSTM", metrics_lstm)

# ============================================
# Step 10: Privacy-Preserving Logistic Regression
# ============================================

dp_model = DPLogisticRegression(epsilon=1.0, data_norm=2.0, max_iter=1000)
dp_model.fit(X_train_scaled, y_train)
y_pred_dp = dp_model.predict(X_test_scaled)

print("\nDifferentially Private Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_dp, zero_division=0))
metrics_dp = evaluate_model(y_test, y_pred_dp)
results['DP Logistic Regression'] = metrics_dp
plot_metrics("DP Logistic Regression", metrics_dp)

# ============================================
# Step 11: Compare Model Performance
# ============================================

results_df = pd.DataFrame(results).T
print("\n=== Model Performance Comparison ===\n")
print(results_df)

# Visualization of performance across all models
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison (with DP Model)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
