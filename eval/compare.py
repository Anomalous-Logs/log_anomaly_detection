import pandas as pd
import dask.dataframe as dd
import numpy as np
import re
import time
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


model_types = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNeighbors', 'XGBoost', 'LightGBM', 'CatBoost']
log_file = 'compare.log'


def read_dataset(filename):
    # Load the dataset
    df = dd.read_csv(filename)
    # df = dd.read_csv('data/CICFlowMeter_out.csv')
    return df.compute()

def read_mapping(filename):
    label_dict = {}
    # Read the mapping file
    with open(filename, 'r') as f:
        for line in f:
            # Use regex to split by one or more tab characters
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2:
                label = parts[0]  # The label (first part)
                value = int(parts[1])  # The corresponding integer value (second part)
                label_dict[label] = value
    return label_dict

def preprocess(df: pd.DataFrame, label_dict = None):
    # Preprocessing
    X = df.drop(columns=['Label'])
    y = df['Label']

    X, y = undersample(X, y, 'Benign')

    X = normalize(X)

    y = apply_mapping(y, label_dict)

    X, y = oversample(X, y)
    
    # X, pca_time = preprocess_pca(X)

    return X, y

def undersample(X, y, value):
    X = pd.concat([X,y], axis=1)
    
    # Seperate majority and minority classes
    df_majority = X[y == value]
    df_minority = X[y != value]
    
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

    return X, y

def apply_mapping(y, label_dict):
    # Map the labels in your target variable
    y = y.map(label_dict)
    return y

def normalize(X):
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
    X = preprocessor.fit_transform(X)
    return X

def oversample(X, y):
    # Oversampling
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    return X, y

def preprocess_pca(X):
    # PCA
    # pca = PCA(n_components='mle', svd_solver='full')  # Specify the number of components
    pca = PCA(n_components=0.9, svd_solver='auto', iterated_power='auto', tol=0.0, whiten=False, random_state=42)  # Specify the number of components
    begin = time.time()
    X_pca = pca.fit_transform(X)
    end = time.time()
    pca_time = end-begin
    
    # Explained variance ratio to understand how much information is retained
    explained_variance = pca.explained_variance_ratio_
    logging.info(f"\nNumber of Components: {pca.n_components_}")
    logging.info(f"\nNumber of Sampless: {pca.n_samples_}")
    logging.info(f"\nFeatures in fit: {pca.n_features_in_}")
    logging.info(f"\nNumber of Components: {pca.n_components_}")
    logging.info(f"PCA time: {pca_time:2f}")
    logging.info(f"Explained variance:\n{explained_variance}")

    return X_pca, pca_time

def preprocess_tsvd(X):
    svd = TruncatedSVD(n_components=100, algorithm='arpack')  # Specify the number of components
    begin = time.time()
    X_svd = svd.fit_transform(X)
    end = time.time()
    svd_time = end-begin
    
    # Explained variance ratio to understand how much information is retained
    explained_variance = svd.explained_variance_ratio_
    logging.info(f"\nNumber of Components: {svd.n_components_}")
    logging.info(f"\nNumber of Sampless: {svd.n_samples_}")
    logging.info(f"\nFeatures in fit: {svd.n_features_in_}")
    logging.info(f"\nNumber of Components: {svd.n_components_}")
    logging.info(f"PCA time: {svd_time:2f}")
    logging.info(f"Explained variance:\n{explained_variance}")

    return X_svd, svd_time


def init_model(model_type):
    if (model_type == 'XGBoost'):
        return XGBClassifier(eval_metric='logloss', random_state=42)
    if (model_type == 'LightGBM'):
        return LGBMClassifier(random_state=42)
    if (model_type == 'LogisticRegression'):
        return LogisticRegression(max_iter=200, random_state=42)
    if (model_type == 'SVM'):
        return SVC(kernel="rbf", max_iter=200, random_state=42)
    if (model_type == 'GaussianNB'):
        return GaussianNB()
    '''
    if (model_type == 'BernoulliNB'):
        return BernoulliNB(random_state=42)
    if (model_type == 'CategoricalNB'):
        return CategoricalNB(random_state=42)
    '''
    # if (model_type == 'ComplementNB'):
        # return ComplementNB()
    if (model_type == 'DecisionTree'):
        return DecisionTreeClassifier(random_state=42)
    if (model_type == 'RandomForest'):
        return RandomForestClassifier(random_state=42)
    if (model_type == 'KNeighbors'):
        return KNeighborsClassifier()
    if (model_type == 'CatBoost'):
        return CatBoostClassifier(random_state=42)
    return None

def train(model, X_train, y_train):
    begin = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    execution_time = end-begin

    logging.info(f"Model fitting time: {execution_time:2f}")

    return model, execution_time

def predict(model, X):
    # Predictions
    begin = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    end = time.time()
    prediction_time = end-begin
    
    return y_pred, y_proba, prediction_time

def cross_validate(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)  # Adjust CV folds as necessary
    logging.info(f"Cross-Validation Score: {scores.mean():.4f}")

def test(model, X_test, y_test):
    y_pred, y_proba, prediction_time = predict(model, X_test)

    # Log loss as a measure of uncertainty
    loss = log_loss(y_test, y_proba)

    # Certainty scores (example using maximum class probability)
    certainty_scores = np.max(y_proba, axis=1)
    
    # Evaluation
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Log Loss: {loss:.4f}")
    logging.info(f"Certainty Scores: {certainty_scores}")
    logging.info(f"Model prediction time: {prediction_time:2f}")

def dump_model(model, model_type):
    # Dump
    joblib.dump(model, f'model/{model_type}.joblib')

def evaluate(X, y, model_type, dumps = True):
    print(f"{model_type}")
    logging.info('\n'+24*"="+f"\n{model_type}\n"+24*"="+'\n')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = init_model(model_type)
    model, execution_time = train(model, X_train, y_train)
    cross_validate(model, X, y)
    test(model, X_test, y_test)

    if dumps:
        dump_model(model, model_type)

if __name__ == '__main__':
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    df = read_dataset('data/CICFlowMeter_out.300000head.csv')
    label_dict = read_mapping('data/Readme.txt')
    X, y = preprocess(df, label_dict)
    for model_type in model_types:
        evaluate(X, y, model_type)
