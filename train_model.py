import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Dataset (Place IRIS.csv in same folder)
df = pd.read_csv("IRIS.csv")

print("Dataset Loaded Successfully")

# Assume last column is species
print("Columns:", df.columns)

# Rename columns if needed (common iris format)
# sepal_length, sepal_width, petal_length, petal_width, species

# Encode species
le = LabelEncoder()
df["species"] = LabelEncoder().fit_transform(df["species"])


# Features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved as model.pkl")
