import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_earthquake_model():

    print("🔄 Loading dataset...")

    # Use python engine (slower but handles all weird CSVs)
    df = pd.read_csv(
        "../data/earthquake_clean.csv",
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    print("📊 Loaded rows:", len(df))

    # Keep required columns
    df = df[["latitude", "longitude", "depth", "mag"]]
    df = df.dropna()

    print("📈 Training data size:", df.shape)

    X = df[["latitude", "longitude", "depth"]]
    y = df["mag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training model...")

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("📉 MSE:", mse)

    joblib.dump(model, "earthquake_model.pkl")
    print("💾 Model saved as earthquake_model.pkl")

if __name__ == "__main__":
    train_earthquake_model()

