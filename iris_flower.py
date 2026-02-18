# ======================================
# TASK 3: IRIS FLOWER CLASSIFICATION
# Dataset: Kaggle - arshid/iris-flower-dataset
# ======================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset (YOUR EXACT PATH)
df = pd.read_csv(r"C:\Kavya\Internship\Codsoft\Data science\iris.csv")

# 2. Display dataset
print(df.head())
print(df.info())

# 3. Separate features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# 4. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
