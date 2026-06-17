import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

np.random.seed(42)
n = 1000

age         = np.random.randint(18, 65, n)
income      = np.random.randint(20000, 120000, n)
credit_score= np.random.randint(300, 850, n)
loan_amount = np.random.randint(1000, 50000, n)
employment_years = np.random.randint(0, 30, n)
num_accounts= np.random.randint(1, 10, n)
num_loans   = np.random.randint(0, 5, n)
monthly_expenses = np.random.randint(500, 5000, n)
savings     = np.random.randint(0, 100000, n)
debt_ratio  = np.round(np.random.uniform(0.0, 1.0, n), 2)

# Target: loan approved (1) or not (0)
approved = (
    (credit_score > 600).astype(int) +
    (income > 50000).astype(int) +
    (debt_ratio < 0.5).astype(int) +
    (employment_years > 2).astype(int) +
    (savings > 10000).astype(int)
)
approved = (approved >= 3).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "employment_years": employment_years,
    "num_accounts": num_accounts,
    "num_loans": num_loans,
    "monthly_expenses": monthly_expenses,
    "savings": savings,
    "debt_ratio": debt_ratio,
    "approved": approved
})

df.to_csv("loan_dataset.csv", index=False)
print("Dataset saved: loan_dataset.csv")
print(df.head())
print(f"\nApproval rate: {df['approved'].mean():.1%}")

X = df.drop("approved", axis=1)
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Model saved: model.pkl")
