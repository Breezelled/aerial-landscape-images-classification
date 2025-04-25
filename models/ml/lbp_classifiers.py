import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)

# === 1. Load CSV ===
df = pd.read_csv("image_label.csv")
df["image"] = df["image"].apply(lambda x: os.path.join("Aerial_Landscapes", x))

# === 2. Define LBP extractor ===
def extract_lbp_features(img_path, P=8, R=1.0, method='uniform'):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((P + 2,))
    lbp = local_binary_pattern(img, P, R, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# === 3. Extract LBP features ===
print("Extracting LBP features...")
features, labels = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    features.append(extract_lbp_features(row["image"]))
    labels.append(row["label"])

X = np.array(features)
y = np.array(labels)

# === 4. PCA ===
X_pca = PCA(n_components=10).fit_transform(X)

# === 5. Encode + split ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === 6. Define models ===
models = {
    "SVM": SVC(probability=True),
    "kNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# === 7. Top-K Accuracy function ===
def top_k_accuracy(y_true, y_proba, k):
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

# === 8. Evaluate and save ===
results = []

with open("lbp_model_detailed_report.txt", "w") as f:
    print("Evaluating LBP-based models...\n")
    f.write("Evaluating LBP-based models...\n\n")

    for name, model in models.items():
        print(f"Training model: {name}")
        f.write(f"\nTraining model: {name}\n")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc1 = accuracy_score(y_test, preds)
        acc3 = top_k_accuracy(y_test, proba, k=3)
        acc5 = top_k_accuracy(y_test, proba, k=5)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        cls_report = classification_report(y_test, preds, target_names=le.classes_)

        print(f"Acc@1: {acc1:.4f}, Acc@3: {acc3:.4f}, Acc@5: {acc5:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(cls_report)

        f.write(f"Acc@1: {acc1:.4f}, Acc@3: {acc3:.4f}, Acc@5: {acc5:.4f}\n")
        f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        f.write(cls_report)
        f.write("\n" + "="*80 + "\n")

        results.append({
            "Model": name,
            "Acc@1": acc1,
            "Acc@3": acc3,
            "Acc@5": acc5,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1
        })

# === 9. Save summary to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("lbp_model_results_with_topk.csv", index=False)
print("\nSaved: lbp_model_results_with_topk.csv")
print("Saved: lbp_model_detailed_report.txt")
