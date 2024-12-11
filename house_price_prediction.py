import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample data
data = {
    'Area (sq ft)': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age (years)': [10, 15, 20, 25, 30],
    'Price ($)': [300000, 400000, 350000, 500000, 450000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Area (sq ft)', 'Bedrooms', 'Age (years)']]
y = df['Price ($)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Test with new data
new_house = pd.DataFrame([[2500, 4, 15]], columns=['Area (sq ft)', 'Bedrooms', 'Age (years)'])
predicted_price = model.predict(new_house)
print(f"Predicted Price for the new house: ${predicted_price[0]:.2f}")
