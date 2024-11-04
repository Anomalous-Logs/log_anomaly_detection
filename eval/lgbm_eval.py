import pandas as pd
import dask.dataframe as dd
import numpy as np
import re
import time
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Load the dataset
df = dd.read_csv('data/CICFlowMeter_out.300000head.csv')
# df = dd.read_csv('data/CICFlowMeter_out.csv')
label_dict = {}

# Read the mapping file
with open('data/Readme.txt', 'r') as f:
    for line in f:
        # Use regex to split by one or more tab characters
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 2:
            label = parts[0]  # The label (first part)
            value = int(parts[1])  # The corresponding integer value (second part)
            label_dict[label] = value

# Preprocessing
X = df.compute()
y = df['Label'].compute()

# Seperate majority and minority classes
df_majority = X[y == 'Benign']
df_minority = X[y != 'Benign']

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                    replace=False,     # sample without replacement
                                    n_samples=len(df_minority),    # to match minority class
                                    random_state=42)  # reproducible results

# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = df_balanced.drop(columns=['Label'])
y = df_balanced['Label']

# Map the labels in your target variable
y_encoded = y.map(label_dict)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X[categorical_cols] = X[categorical_cols].astype(str)  # Convert to string if not already

# Use ColumnTransformer to handle different types of preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), X.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label_encoder', OneHotEncoder())
        ]), categorical_cols)
    ]
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Oversampling
smote = SMOTE(random_state=42)
X_processed, y_encoded = smote.fit_resample(X_processed, y_encoded)

n_components = range(1, 64)
cv_scores = []

for n in n_components:
    # PCA
    pca = PCA(n_components=n)  # Specify the number of components
    begin = time.time()
    X_pca = pca.fit_transform(X_processed)
    end = time.time()
    pca_time = end-begin
    
    # Explained variance ratio to understand how much information is retained
    explained_variance = pca.explained_variance_ratio_
    print("\nNumber of Components:", n)
    print(f"PCA time: {pca_time:2f}")
    print("Explained variance:")
    print(explained_variance)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train the LightGBM classifier
    model = LGBMClassifier(random_state=42)
    begin = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    execution_time = end-begin
    
    # Predictions
    begin = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    end = time.time()
    prediction_time = end-begin
    
    scores = cross_val_score(model, X_pca, y_encoded, cv=5)  # Adjust CV folds as necessary
    cv_scores.append(scores.mean())

    # Log loss as a measure of uncertainty
    loss = log_loss(y_test, y_proba)

    # Certainty scores (example using maximum class probability)
    certainty_scores = np.max(y_proba, axis=1)
    
    # Evaluation
    print(f"Cross-Validation Score: {scores.mean():.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Log Loss: {loss:.4f}")
    print("Certainty Scores:", certainty_scores)
    print(f"Model fitting time: {execution_time:2f}")
    print(f"Model prediction time: {prediction_time:2f}")

    # Dump
    joblib.dump(model, f'model/lgbm{n}.joblib')

# Optionally print all scores at once
print("\nCross-Validation Scores:")
for n, score in zip(n_components_range, cv_scores):
    print(f"Number of Components: {n}, Score: {score:.4f}")
