# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import shap
import joblib

# --- 2. Load Dataset ---
df = pd.read_csv("books_dataset.csv")  # make sure CSV is in same folder
df = df.dropna(subset=['price', 'original_price', 'category', 'brand'])

# --- 3. Prepare Features & Target ---
# Drop "cheat" column
features = ['original_price', 'rating', 'rating_count', 'category', 'brand', 'availability']
target = 'price'

# Encode categorical features and save encoders
encoders = {}
for col in ['category', 'brand', 'availability']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Model ---
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# --- 5. Evaluate Model ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Compute RMSE manually
r2 = r2_score(y_test, y_pred)

print("=== Evaluation Metrics ===")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- 6. SHAP Analysis ---
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# --- 7. Save Model and Encoders ---
joblib.dump(model, "xgb_product_model.joblib")
joblib.dump(encoders, "label_encoders.joblib")
print("Model and encoders saved successfully!")
