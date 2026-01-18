import json
import pickle
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # Changed from LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load dataset
# =========================
with open("data/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for item in data["sentences"]:
    if item.get("training", True):
        texts.append(item["text"].strip())
        labels.append(item["intent"])

# =========================
# 2. Remove rare intents (<2 samples)
# =========================
label_counts = Counter(labels)
filtered_texts = []
filtered_labels = []

for text, label in zip(texts, labels):
    # slightly increased threshold helps stability, but keeping 2 is fine for now
    if label_counts[label] >= 2:
        filtered_texts.append(text)
        filtered_labels.append(label)

print(f"Total intents kept: {len(set(filtered_labels))}")

# =========================
# 3. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    filtered_texts,
    filtered_labels,
    test_size=0.2,
    random_state=42,
    stratify=filtered_labels
)

# =========================
# 4. TF-IDF (CRITICAL FIXES HERE)
# =========================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,      # <--- CHANGED: "up", "down", "no" are vital!
    ngram_range=(1, 2),   # Unigrams and Bigrams
    max_features=5000,
    sublinear_tf=True     # <--- ADDED: Scales counts to log, helps with frequency spikes
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. Model (Switched to SVM)
# =========================
# LinearSVC is often superior for small-dataset text classification
model = LinearSVC(
    class_weight='balanced', # <--- ADDED: Heavily penalizes mistakes on rare classes
    random_state=42,
    dual="auto"              # Handles n_samples < n_features better
)

model.fit(X_train_vec, y_train)

# =========================
# 6. Evaluation
# =========================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# 7. Save
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")