# Comprehensive phishing-detection pipeline (Steps 1-5 fixed and extended)
# Written in British English. No icons in the code.
# Ensure you have 'Cleaned_PhishingEmailData.csv' in the working directory.

# -------------------------
# Step 0: Environment and imports
# -------------------------
import os
# Suppress oneDNN informational messages from TensorFlow early
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import time
import random
import math

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier

from diffprivlib.models import LogisticRegression as DPLogisticRegression

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PyTorch / Hugging Face / sentence-transformers
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertModel
from sentence_transformers import SentenceTransformer

# Utilities
from scipy.sparse import hstack

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# -------------------------
# Step 1: Load and clean data
# -------------------------
df = pd.read_csv('Cleaned_PhishingEmailData.csv', encoding='ISO-8859-1')

# Basic cleaning
df['Email_Content'] = df['Email_Content'].fillna("").astype(str)
df['Email_Subject'] = df['Email_Subject'].fillna("").astype(str)

# Ensure label is 0/1
df['Label'] = df['Label'].apply(lambda x: 1 if int(x) == 1 else 0)

def clean_text(text):
    """Lowercase, replace urls and emails with tokens and remove non-alphanumeric characters (preserve spaces)."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_content'] = df['Email_Content'].apply(clean_text)
df['clean_subject'] = df['Email_Subject'].apply(clean_text)

# -------------------------
# Step 2: Feature engineering & EDA
# -------------------------
df['subject_length'] = df['Email_Subject'].apply(lambda x: len(str(x)))
df['link_count'] = df['Email_Content'].apply(lambda x: len(re.findall(r"http\S+", str(x))))
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0)
df['num_exclamations'] = df['Email_Content'].apply(lambda x: str(x).count('!'))
df['num_question'] = df['Email_Content'].apply(lambda x: str(x).count('?'))
df['num_digits'] = df['Email_Content'].apply(lambda x: sum(c.isdigit() for c in str(x)))

# EDA plots
plt.figure(figsize=(8,4))
sns.histplot(df['subject_length'], bins=30, kde=True)
plt.title('Distribution of Subject Lengths')
plt.xlabel('Subject length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='has_link', data=df)
plt.title('Presence of Links in Emails')
plt.xlabel('Has link')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Label', data=df)
plt.title('Label Distribution (0 = Legit, 1 = Phishing)')
plt.tight_layout()
plt.show()

# -------------------------
# Step 3: Clustering & grouping (six algorithms)
# -------------------------
cluster_features = ['subject_length', 'link_count', 'has_link', 'num_exclamations', 'num_question', 'num_digits']
X_cluster = df[cluster_features].fillna(0).values

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_cluster)

clustering_algorithms = {
    'KMeans': KMeans(n_clusters=2, random_state=SEED),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=2),
    'MeanShift': MeanShift(),
    'Spectral': SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=SEED),
    'GMM': GaussianMixture(n_components=2, random_state=SEED)
}

cluster_results = {}
for name, algorithm in clustering_algorithms.items():
    try:
        labels = algorithm.fit_predict(X_cluster)
    except Exception as exc:
        print(f"{name} failed: {exc}")
        continue
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='Set2', legend='brief')
    plt.title(f"{name} Clustering (PCA projection)")
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.tight_layout()
    plt.show()
    ari = adjusted_rand_score(df['Label'].values, labels)
    cluster_results[name] = {'labels': labels, 'ARI': ari}
    print(f"{name}: Adjusted Rand Index vs true labels = {ari:.4f}")

# -------------------------
# Step 4: Train/test split and scaling (no leakage)
# -------------------------
numeric_features = ['subject_length', 'link_count', 'has_link', 'num_exclamations', 'num_question', 'num_digits']
X_numeric = df[numeric_features].fillna(0).values
y = df['Label'].values

# Stratified split to maintain class distribution
X_num_train, X_num_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_numeric, y, np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# Text splits (train-only fitting for tokenisers)
texts = df['clean_content'].values
texts_train = texts[idx_train]
texts_test = texts[idx_test]

subjects = df['clean_subject'].values
subjects_train = subjects[idx_train]
subjects_test = subjects[idx_test]

# -------------------------
# Step 5: Models and evaluation utilities
#   - Logistic Regression + TF-IDF
#   - LightGBM + sentence-transformer embeddings
#   - DistilBERT features + Logistic Regression
#   - BERT fine-tuning (PyTorch)
# Also LSTM (Keras) as text baseline
# -------------------------
def evaluate_and_report(name, y_true, y_pred, y_score=None):
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_score) if y_score is not None else roc_auc_score(y_true, y_pred)
    except Exception:
        roc = None
    print(f"{name} summary: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC-AUC': roc}

results = {}

# ---- Model 1: Logistic Regression with TF-IDF ----
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
tfidf.fit(texts_train)  # fit on train only
X_tfidf_train = tfidf.transform(texts_train)
X_tfidf_test = tfidf.transform(texts_test)

X_lr_train = hstack([X_tfidf_train, np.array(X_num_train_scaled)])
X_lr_test = hstack([X_tfidf_test, np.array(X_num_test_scaled)])

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
lr.fit(X_lr_train, y_train)
y_pred_lr = lr.predict(X_lr_test)
y_score_lr = lr.predict_proba(X_lr_test)[:,1]
results['Logistic TFIDF'] = evaluate_and_report('Logistic Regression (TF-IDF)', y_test, y_pred_lr, y_score_lr)

# ---- Model 2: LightGBM on sentence-transformer embeddings + numeric features ----
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

X_embed_train = embedder.encode(list(texts_train), show_progress_bar=True, convert_to_numpy=True)
X_embed_test = embedder.encode(list(texts_test), show_progress_bar=True, convert_to_numpy=True)

X_lgb_train = np.hstack([X_embed_train, X_num_train_scaled])
X_lgb_test = np.hstack([X_embed_test, X_num_test_scaled])

lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=SEED)
lgb_model.fit(X_lgb_train, y_train)
y_pred_lgb = lgb_model.predict(X_lgb_test)
y_score_lgb = lgb_model.predict_proba(X_lgb_test)[:,1]
results['LightGBM Embeddings'] = evaluate_and_report('LightGBM (Embeddings)', y_test, y_pred_lgb, y_score_lgb)

# ---- Model 3: DistilBERT feature extraction + Logistic Regression ----
distil_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distil_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
distil_model.eval()

def encode_distilbert(texts_list, batch_size=32):
    embeddings = []
    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i+batch_size]
        enc = distil_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        input_ids = enc['input_ids'].to(DEVICE)
        attention_mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            out = distil_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            summed = (last_hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            pooled = summed / counts
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

X_distil_train = encode_distilbert(list(texts_train), batch_size=32)
X_distil_test = encode_distilbert(list(texts_test), batch_size=32)

X_distil_train_comb = np.hstack([X_distil_train, X_num_train_scaled])
X_distil_test_comb = np.hstack([X_distil_test, X_num_test_scaled])

clf_distil = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
clf_distil.fit(X_distil_train_comb, y_train)
y_pred_distil = clf_distil.predict(X_distil_test_comb)
y_score_distil = clf_distil.predict_proba(X_distil_test_comb)[:,1]
results['DistilBERT + LR'] = evaluate_and_report('DistilBERT features + Logistic Regression', y_test, y_pred_distil, y_score_distil)

# ---- Model 4: BERT fine-tuning (PyTorch). Will fallback if insufficient resources ----
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
try:
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)
    # Simple PyTorch dataset and dataloader
    from torch.utils.data import Dataset, DataLoader

    class SimpleTextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset_bert = SimpleTextDataset(texts_train, y_train, bert_tokenizer, max_len=128)
    test_dataset_bert = SimpleTextDataset(texts_test, y_test, bert_tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset_bert, batch_size=32, shuffle=False)

    def train_bert(model, train_loader, epochs=2, lr=2e-5):
        optim = AdamW(model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            start = time.time()
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_b = batch['labels'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_b)
                loss = outputs.loss
                loss.backward()
                optim.step()
                scheduler.step()
                epoch_loss += loss.item()
            end = time.time()
            print(f"BERT epoch {epoch+1}/{epochs}: loss={epoch_loss/len(train_loader):.4f}, time={end-start:.1f}s")
        return model

    # Run a short fine-tuning if GPU is available, otherwise do a light pass
    epochs_to_run = 2 if torch.cuda.is_available() else 1
    bert_model = train_bert(bert_model, train_loader, epochs=epochs_to_run, lr=2e-5)

    # Evaluate
    bert_model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = np.argmax(logits.cpu().numpy(), axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
    results['BERT Fine-tuned'] = evaluate_and_report('BERT Fine-tuned', y_test, np.array(all_preds), np.array(all_probs))

except Exception as exc:
    print("BERT fine-tuning skipped or failed due to:", exc)
    print("Proceeding with other models.")

# ---- Baseline: LSTM (Keras) on train-only tokeniser ----
# Fit tokenizer on training texts only to avoid leakage
keras_tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
keras_tokenizer.fit_on_texts(list(texts_train))

seq_train = keras_tokenizer.texts_to_sequences(list(texts_train))
seq_test = keras_tokenizer.texts_to_sequences(list(texts_test))

X_train_lstm = pad_sequences(seq_train, maxlen=100)
X_test_lstm = pad_sequences(seq_test, maxlen=100)

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train with few epochs for quick demonstration
lstm_model.fit(X_train_lstm, y_train, epochs=3, batch_size=64, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
results['LSTM'] = evaluate_and_report('LSTM', y_test, y_pred_lstm)

# -------------------------
# Step 6: Privacy-preserving experiments (DP logistic regression)
# -------------------------
epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
dp_results = {}
for eps in epsilons:
    try:
        dp = DPLogisticRegression(epsilon=eps, data_norm=2.0, max_iter=1000, random_state=SEED)
        dp.fit(X_num_train_scaled, y_train)
        y_pred_dp = dp.predict(X_num_test_scaled)
        res = evaluate_and_report(f"DP Logistic Regression (epsilon={eps})", y_test, y_pred_dp)
        dp_results[eps] = res['Accuracy']
    except Exception as exc:
        print(f"DP model failed for epsilon={eps}: {exc}")
        dp_results[eps] = None

# Plot privacy-utility trade-off
plt.figure(figsize=(7,4))
eps_vals = [k for k in dp_results.keys() if dp_results[k] is not None]
acc_vals = [dp_results[k] for k in eps_vals]
if len(eps_vals) > 0:
    plt.plot(eps_vals, acc_vals, marker='o')
    plt.xscale('log')
    plt.xlabel('Epsilon (log scale; higher = less privacy)')
    plt.ylabel('Accuracy')
    plt.title('Privacy–Utility Trade-off (DP Logistic Regression)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# Step 7: Consolidated results and plots
# -------------------------
results_df = pd.DataFrame(results).T
print("\n=== Model Performance Comparison ===\n")
print(results_df)

plt.figure(figsize=(10,6))
if 'Accuracy' in results_df.columns and 'F1 Score' in results_df.columns:
    results_df[['Accuracy','F1 Score']].plot(kind='bar', rot=45)
    plt.title('Model comparison: Accuracy and F1 Score')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

results_df.to_csv('model_comparison_results.csv', index=True)

# -------------------------
# Finished
# -------------------------
print("\nPipeline complete.")
print("- Tokenisers and vectorizers were fitted only on training data to avoid leakage.")
print("- Clustering ARI values (above) show how unsupervised clusters align with labels; interpret with caution.")
print("- DistilBERT and sentence-transformer embeddings provide compact, fast features for downstream classifiers.")
print("- BERT fine-tuning was attempted; it may be skipped on systems without GPU.")
print("- DP experiments ran a small epsilon sweep and produced a privacy–utility plot.")
print("Results saved to 'model_comparison_results.csv'.")
