import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Path to IMDB dataset (TSV format)
IMDB_CSV = Path(r"data/labeledTrainData.tsv")

# Load dataset
df = pd.read_csv(IMDB_CSV, sep="\t")
df = df.dropna(subset=["review", "sentiment"])
texts = df["review"].astype(str).tolist()
labels = (df["sentiment"].str.lower() == "positive").astype(int).to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF vectorization
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
X_train = vec.fit_transform(X_train).astype(np.float32).toarray()
X_test = vec.transform(X_test).astype(np.float32).toarray()

# Torch datasets
train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)
test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=256, shuffle=False)

# Model
input_dim = X_train.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(3):
    model.train(); correct=0; total=0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        pred = logits.argmax(1)
        correct += (pred==yb).sum().item(); total += yb.size(0)
    print(f"Epoch {epoch+1}/3 - Train acc: {correct/total:.3f}")

# Evaluation
model.eval(); preds=[]; gold=[]
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        gold.extend(yb.cpu().numpy().tolist())

print("Test Accuracy:", accuracy_score(gold, preds))
print(classification_report(gold, preds, target_names=["negative","positive"]))
