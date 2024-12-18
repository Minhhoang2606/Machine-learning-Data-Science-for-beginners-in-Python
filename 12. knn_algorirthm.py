'''
KNN Algorirthm with Wine Dataset
Author: Henry Ha
'''
# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

#TODO: Load and prepare the dataset

# Load the Wine Dataset
# Load the Wine Dataset
wine_data = load_wine(as_frame=True)
df = wine_data.frame  # Convert to DataFrame

# Display the first few rows
print(df.head())


# Inspect the dataset
print("Dataset Keys:", wine_data.keys())
print("Feature Names:", wine_data.feature_names)
print("Target Classes:", wine_data.target_names)

# Check for missing values and data types
print(df.info())

# Summary statistics
print(df.describe())

#TODO Exploratory Data Analysis (EDA)

# Feature distribution
# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(15, 12))

# Plot distributions of all features
for i, col in enumerate(df.columns[:-1]):  # Exclude 'target'
    plt.subplot(4, 4, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Correlation between features
# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Subset of features for visualization
selected_features = ['alcohol', 'color_intensity', 'flavanoids', 'hue', 'proline']

# Explicitly set 3 distinct colors for the target classes
custom_palette = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # Blue, Orange, Green

# Pair plot with hue for target variable
sns.pairplot(df[selected_features + ['target']], hue='target', palette=custom_palette)
plt.show()

# Target Variable Distribution
# Count plot for the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='pastel')
plt.title("Distribution of Target Classes")
plt.xlabel("Wine Classes")
plt.ylabel("Count")
plt.show()

#TODO Train-Test Split and Feature Scaling

# Split the data
# Define input features (X) and target labels (y)
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Feature scaling
# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify scaling
import pandas as pd
print("Scaled Training Features:\n", pd.DataFrame(X_train_scaled, columns=X.columns).describe())

#TODO Building and Training the KNN Model

# Import and Initialize the Model
# Initialize the KNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Predict target labels for the test set
y_pred = knn.predict(X_test_scaled)

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print overall accuracy
print("Accuracy Score:", accuracy_score(y_test, y_pred))
