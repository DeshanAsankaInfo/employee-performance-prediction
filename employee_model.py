import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
dataset_path = "C:/Users/wwwde/PycharmProjects/EmpModel/data/Employee Attrition.csv"
df = pd.read_csv(dataset_path)

# View the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Simulate the missing target columns if they do not exist
if 'performance' not in df.columns:
    df['performance'] = np.random.uniform(0.5, 1.0, df.shape[0])  # Simulate performance score between 0.5 and 1.0
if 'promotion_likelihood' not in df.columns:
    df['promotion_likelihood'] = np.random.uniform(0, 1, df.shape[0])  # Simulate promotion likelihood between 0 and 1
if 'attrition_risk' not in df.columns:
    df['attrition_risk'] = np.random.randint(0, 2, df.shape[0])  # Simulate attrition risk as binary (0 or 1)

# Drop rows where target variables are NaN
df = df.dropna(subset=['satisfaction_level', 'performance', 'promotion_likelihood', 'attrition_risk'])

# Feature selection (independent variables)
X = df[['last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'salary']]

# Target variables (multi-output)
y = df[['satisfaction_level', 'performance', 'promotion_likelihood', 'attrition_risk']]

# Convert categorical 'salary' into numerical values (low = 0, medium = 1, high = 2)
X.loc[:, 'salary'] = X['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# Split the dataset into training and testing sets (for multi-output regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model with multi-output capability
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Train the multi-output model
model.fit(X_train, y_train)

# Test the model and get predictions
y_pred = model.predict(X_test)

# Evaluate the model for each output
for i, col in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col} - Mean Squared Error: {mse}")
    print(f"{col} - R² Score: {r2}")

# Save the model with compression
joblib.dump(model, 'employee_multioutput_model.pkl', compress=3)

# Load the saved model
loaded_model = joblib.load('employee_multioutput_model.pkl')

# Convert new data into a DataFrame with the correct feature names
new_data = pd.DataFrame([[0.8, 5, 150, 3, 1]], columns=['last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'salary'])

# Predict multiple outcomes for the new data
predicted_outcomes = loaded_model.predict(new_data)

# Display the predictions
predicted_satisfaction, predicted_performance, predicted_promotion, predicted_attrition_risk = predicted_outcomes[0]
print(f"Predicted Satisfaction: {predicted_satisfaction}")
print(f"Predicted Performance: {predicted_performance}")
print(f"Predicted Promotion Likelihood: {predicted_promotion}")
print(f"Predicted Attrition Risk: {predicted_attrition_risk}")
