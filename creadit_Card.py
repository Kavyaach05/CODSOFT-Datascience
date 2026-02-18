# ===============================
# CREDIT CARD FRAUD DETECTION ML
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# ===============================
# 1. LOAD DATASET
# ===============================
# Dataset should have a column named 'Class'
# 0 = Genuine, 1 = Fraud

data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print(data['Class'].value_counts())

# ===============================
# 2. SPLIT FEATURES & LABEL
# ===============================
X = data.drop("Class", axis=1)
y = data["Class"]

# ===============================
# 3. NORMALIZE DATA
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:", np.bincount(y_train))

# ===============================
# 5. HANDLE CLASS IMBALANCE (SMOTE)
# ===============================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train_sm))

# ===============================
# 6. TRAIN MODELS
# ===============================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sm, y_train_sm)

# ===============================
# 7. PREDICTIONS
# ===============================
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# ===============================
# 8. EVALUATION FUNCTION
# ===============================
def evaluate_model(name, y_test, pred):
    print("\n======", name, "======")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

# ===============================
# 9. RESULTS
# ===============================
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
