import pandas as pd
from sklearn.model_selection import train_test_split
from model import (
    train_simple_linear_regression,
    train_multiple_linear_regression,
    evaluate_model,
)

print("Sales Revenue Prediction System")

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("data/sales_data.csv")

# ----------------------------
# 2. Define Features (X) and Target (y)
# ----------------------------
X = data.iloc[:, :-1].values  # All columns except last
y = data.iloc[:, -1].values   # Last column (Sales)

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Simple Linear Regression
# ----------------------------
simple_model = train_simple_linear_regression(X_train, y_train)

r2_simple, mse_simple, rmse_simple, y_pred_simple = evaluate_model(
    simple_model, X_test[:, [0]], y_test
)

print("\n--- Simple Linear Regression ---")
print("R2 Score:", r2_simple)
print("MSE:", mse_simple)
print("RMSE:", rmse_simple)
print("Coefficient:", simple_model.coef_)
print("Intercept:", simple_model.intercept_)

# ----------------------------
# 5. Multiple Linear Regression
# ----------------------------
multiple_model = train_multiple_linear_regression(X_train, y_train)

r2_multi, mse_multi, rmse_multi, y_pred_multi = evaluate_model(
    multiple_model, X_test, y_test
)

print("\n--- Multiple Linear Regression ---")
print("R2 Score:", r2_multi)
print("MSE:", mse_multi)
print("RMSE:", rmse_multi)
print("Coefficients:", multiple_model.coef_)
print("Intercept:", multiple_model.intercept_)

# ----------------------------
# 6. Save Predictions to CSV
# ----------------------------
pred_df = pd.DataFrame({
    "Actual_Sales": y_test,
    "Predicted_Sales": y_pred_multi
})

pred_df.to_csv("reports/predictions.csv", index=False)

print("\nPredictions saved to reports/predictions.csv")