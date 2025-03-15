import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ta import add_all_ta_features  # Technical analysis library
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer # Import SimpleImputer

# Load dataset
file_path = "/content/NSE-TATAGLOBAL.csv" #here i have used NSE TATA Global, but you can use any other code
columns = ["Date", "Open", "High", "Low", "Last", "Close", "Total Trade Quality", "Turnover(Lacs)"]
df = pd.read_csv(file_path, names=columns, header=None, skiprows=1)

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # Ensure chronological order
df = dropna(df)  # Remove missing values
df['Turnover(Lacs)'] = df['Turnover(Lacs)'].astype(float)

# Add technical indicators
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Total Trade Quality")

# Define target variable: 1 if next day's close price is higher, else 0
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop unnecessary columns
df = df.drop(columns=["Date", "Last"])

# Impute missing values using SimpleImputer 
# before applying StandardScaler and SMOTE
imputer = SimpleImputer(strategy='mean') # Create an imputer instance
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1]) # Impute missing values

# Standardize features
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Handle imbalanced data using SMOTE
X, y = df.drop(columns=["Target"]), df["Target"]
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=400, max_depth=25, min_samples_split=3, min_samples_leaf=1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Adjust threshold to balance accuracy (between 0.6-0.8)
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_probs > 0.6).astype(int)
print("Adjusted Accuracy:", accuracy_score(y_test, y_pred_adjusted))
print(classification_report(y_test, y_pred_adjusted))

# Predict future trend
future_prediction = model.predict(X.iloc[-1:].values.reshape(1, -1))[0]
print("Future Prediction (1 = Up, 0 = Down):", future_prediction)

# Visualize closing price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], bins=30, kde=True, color='blue')
plt.title("Closing Price Distribution")
plt.xlabel("Closing Price (INR)")
plt.ylabel("Frequency")
plt.show()

# Display profit/loss matrix
plt.figure(figsize=(6, 5))
sns.heatmap(pd.crosstab(y_test, y_pred_adjusted, rownames=["Actual"], colnames=["Predicted"]), annot=True, fmt='d', cmap='coolwarm')
plt.title("Profit/Loss Matrix")
plt.show()
