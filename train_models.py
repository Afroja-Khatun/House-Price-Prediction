import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor

# Load Dataset
df = pd.read_csv("kc_house_data.csv")

# Feature Engineering
df["sales_year"] = pd.to_datetime(df["date"]).dt.year
df["house_age"] = df["sales_year"] - df["yr_built"]
df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

# Features
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_basement",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "house_age",
    "is_renovated"
]

X = df[features]
y = df["price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
xgb.fit(X_train, y_train)

# Evaluation
models = {
    "Linear Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb
}

best_score = -999
best_model = None

for name, model in models.items():

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    rmse = mean_squared_error(
        y_test,
        test_pred
    ) ** 0.5

    print("\n========================")
    print(name)
    print("========================")

    print("Train R2 :", round(train_r2, 4))
    print("Test R2  :", round(test_r2, 4))
    print("RMSE     :", round(rmse, 2))

    if train_r2 - test_r2 > 0.10:
        print("Possible Overfitting")
    else:
        print("No Significant Overfitting")

    if test_r2 > best_score:
        best_score = test_r2
        best_model = model

# Save Best Model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nSuccess!")
print("house_price_model.pkl created successfully")