import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
data = pd.read_csv('restaurant_data.csv')
print("Initial Data Preview:")
print(data.head())
print("\nColumns:", data.columns.tolist())
drop_cols = ['Restaurant ID', 'Restaurant Name', 'Address']
for col in drop_cols:
    if col in data.columns:
        data = data.drop(col, axis=1)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
categorical_cols = data.select_dtypes(include='object').columns.tolist()
if 'Aggregate rating' in categorical_cols:
    categorical_cols.remove('Aggregate rating')  
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data.drop('Aggregate rating', axis=1)
y = data['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n--- {model_name} Evaluation ---")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree Regression")
feature_importance = pd.Series(lr_model.coef_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print("\nTop 10 Important Features (Linear Regression):")
print(feature_importance.head(10))
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label='Linear Regression')
plt.scatter(y_test, y_pred_dt, alpha=0.6, label='Decision Tree', color='red')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show(block=True)
input("Press Enter to exit...")