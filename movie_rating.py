# ======================================
# TASK 2: MOVIE RATING PREDICTION
# ======================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset (IMPORTANT: encoding fix)
df = pd.read_csv(
    r"C:\Kavya\Internship\Codsoft\Data science\IMDb Movies India.csv",
    encoding='latin1'
)

# 2. Display info
print(df.head())
print(df.info())

# 3. Select required columns (CORRECT NAMES)
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# 4. Handle missing values
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')

# Remove rows where Rating is missing (important for regression)
df = df.dropna(subset=['Rating'])

# 5. Encode categorical columns
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor.fit_transform(df['Actor 1'])

# 6. Split features and target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Model Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
