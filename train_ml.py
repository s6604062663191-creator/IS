import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error

# =========================
# LOAD DATA
# =========================
ratings = pd.read_csv("data/u.data", sep="\t",
                      names=["user", "movie", "rating", "timestamp"])

movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
movies = movies[[0] + list(range(5, 24))]
movies.columns = ["movie"] + [f"genre_{i}" for i in range(19)]

df = pd.merge(ratings, movies, on="movie")

print("✅ Data loaded")

# =========================
# FEATURES
# =========================
X = df[["user", "movie"] + [f"genre_{i}" for i in range(19)]]
y = df["rating"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data prepared")

# =========================
# 🔥 OPTIMIZED MODELS (เบา + เร็ว)
# =========================

rf = RandomForestRegressor(
    n_estimators=40,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    n_estimators=40,
    max_depth=3,
    random_state=42
)

et = ExtraTreesRegressor(
    n_estimators=40,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# =========================
# ENSEMBLE
# =========================
model = VotingRegressor([
    ("rf", rf),
    ("gb", gb),
    ("et", et)
])

print("🚀 Training model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"📊 RMSE: {rmse:.4f}")

# =========================
# SAVE MODEL (compressed)
# =========================
joblib.dump(model, "ml_model.pkl", compress=3)

print("✅ Model saved as ml_model.pkl")