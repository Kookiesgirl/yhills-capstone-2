# yhills-capstone-2
MYSELF VIVISHA CATHERIN, I HAVE DEVELOPED THE GIVEN CAPSTONE PROJECT-2

HOUSE PRICE PREDICTION

📌 Code Explanation:

1️⃣ Importing Necessary Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
✔ Pandas & NumPy → Data handling
✔ Matplotlib & Seaborn → Data visualization

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE
✔ Scikit-learn tools → Preprocessing, Model Training, Evaluation
✔ XGBoost → More powerful regression model
✔ Recursive Feature Elimination (RFE) → Feature selection

2️⃣ Load & Explore the Dataset
python
Copy
Edit
df = pd.read_csv("house_prices.csv")
print(df.head())
print(df.info())
✔ Reads the dataset and prints the first few rows & column info
✔ Helps in identifying missing values & data types

3️⃣ Handling Missing Values
python
Copy
Edit
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numerical with median
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical with mode
✔ Numerical Columns → Replaces NaN with the median (reduces the effect of outliers)
✔ Categorical Columns → Replaces NaN with the mode (most frequent value)

4️⃣ Encoding Categorical Variables
python
Copy
Edit
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
✔ Converts categorical variables into numerical values
✔ Uses Label Encoding (e.g., "New York", "San Francisco" → 0, 1)

💡 Potential Improvement: Use One-Hot Encoding instead of Label Encoding if categorical features are non-ordinal.

python
Copy
Edit
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
✔ Helps avoid misinterpretation of numerical values for categories

5️⃣ Define Features (X) and Target (y)
python
Copy
Edit
X = df.drop(columns=["Price"])  # Features
y = df["Price"]  # Target variable
✔ X contains all features except "Price"
✔ y is the target variable (house price)

💡 Improvement: If the dataset has different column names, ensure "Price" is correct. Use:

python
Copy
Edit
print(df.columns)  # Check column names
6️⃣ Splitting the Dataset
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
✔ Splits the data into 80% training, 20% testing
✔ Ensures randomization for fairness

7️⃣ Feature Scaling (Normalization)
python
Copy
Edit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
✔ Standardizes numerical features to mean = 0, standard deviation = 1
✔ Prevents models from being biased toward larger values

8️⃣ Train Multiple Regression Models
python
Copy
Edit
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}
✔ Trains three different models to compare performance:

Linear Regression → Simple model for understanding trends
Decision Tree → Handles non-linear relationships
XGBoost → Advanced boosting model for better accuracy
python
Copy
Edit
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nModel: {name}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
✔ Evaluates models using:

MAE (Mean Absolute Error) → Lower is better
RMSE (Root Mean Squared Error) → Lower is better
R² Score (Coefficient of Determination) → Closer to 1 is better
💡 Improvement: Add cross-validation for better model robustness:

python
Copy
Edit
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R²:", np.mean(scores))
9️⃣ Feature Selection using Recursive Feature Elimination (RFE)
python
Copy
Edit
selector = RFE(LinearRegression(), n_features_to_select=5)
selector.fit(X_train, y_train)
selected_features = X.columns[selector.support_]

print("\nTop 5 Selected Features:", selected_features)
✔ Selects the 5 most important features based on their predictive power
✔ Helps in reducing overfitting & improving model interpretability

💡 Improvement: Visualize feature importance:

python
Copy
Edit
feature_importance = pd.Series(selector.ranking_, index=X.columns)
feature_importance.sort_values().plot(kind="barh", title="Feature Importance")
plt.show()
🚀 Final Summary
✅ Loads & Prepares Data (Handles missing values, encodes categorical features)
✅ Scales Features (Standardization for better model training)
✅ Trains 3 Models (Linear Regression, Decision Tree, XGBoost)
✅ Evaluates Performance (MAE, RMSE, R² Score)
✅ Selects Important Features (Using RFE)

🔄 Next Steps for Improvement
🚀 Hyperparameter Tuning → Optimize Decision Tree & XGBoost with GridSearchCV
🚀 Handle Categorical Data Better → Use One-Hot Encoding instead of Label Encoding
🚀 Feature Engineering → Create new meaningful features (e.g., "Price per Sq. Ft.")
🚀 Deploy Model → Use Flask or Streamlit for a web-based prediction tool



