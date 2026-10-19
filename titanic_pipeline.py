"""
titanic_pipeline.py - Titanic Disaster Survival Prediction Machine Learning Pipeline
Author: Javad Sajadi
Repository: javad-sajadi/Taitanic

Features:
  - Automated feature engineering: Title extraction, family grouping, imputation.
  - Model benchmarking: Random Forest, Support Vector Machine (SVM), Logistic Regression.
  - Stratified K-Fold Cross-Validation and test prediction generation.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_and_preprocess_data(data_dir: str = "titanic"):
    """Loads and preprocesses Titanic training and testing datasets."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        train_path = "train.csv"
        test_path = "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train = len(train_df)
    df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

    # 1. Title Extraction
    df['Title'] = df['Name'].str.extract(r'([A-Za-z]+)\.', expand=True)
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')

    # 2. Family Size Features
    df['FamilySize'] = df['Parch'].fillna(0) + df['SibSp'].fillna(0) + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)

    # 3. Impute Age by Title median
    age_medians = df.groupby('Title')['Age'].median()
    df['Age'] = df.apply(lambda r: age_medians[r['Title']] if pd.isna(r['Age']) else r['Age'], axis=1)

    # 4. Impute Fare by Pclass median
    fare_medians = df.groupby('Pclass')['Fare'].median()
    df['Fare'] = df.apply(lambda r: fare_medians[r['Pclass']] if pd.isna(r['Fare']) else r['Fare'], axis=1)

    # 5. Sex encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0).astype(int)

    # 6. Embarked imputation & one-hot encoding
    df['Embarked'] = df['Embarked'].fillna('S')
    df = pd.get_dummies(df, columns=['Embarked', 'Title', 'Pclass'], drop_first=True)

    # 7. Drop unused columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Split back into train and test
    train_clean = df.iloc[:n_train].copy()
    test_clean = df.iloc[n_train:].copy()

    X_train = train_clean.drop(columns=['Survived'])
    y_train = train_clean['Survived'].astype(int)
    X_test = test_clean.drop(columns=['Survived'])

    return X_train, y_train, X_test, test_df.get('PassengerId')


def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series):
    """Evaluates multiple classification algorithms using 5-fold cross validation."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Support Vector Machine (RBF)": SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n--- 5-Fold Stratified Cross-Validation Benchmark ---")
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled if "Forest" not in name else X_train, y_train, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = mean_acc
        print(f"{name:30s}: Mean Accuracy = {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Performing Model: {best_model_name} ({results[best_model_name] * 100:.2f}%)")
    return models[best_model_name], scaler


def generate_submission(model, scaler, X_train, y_train, X_test, passenger_ids, output_path: str = "submission.csv"):
    """Fits the best model on full training set and writes test predictions."""
    if "Forest" in type(model).__name__:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    else:
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        predictions = model.predict(X_test_s)

    if passenger_ids is not None:
        sub = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions
        })
        sub.to_csv(output_path, index=False)
        print(f"Submission saved to: {output_path} ({len(sub)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML Survival Prediction Pipeline")
    parser.add_argument("--data-dir", type=str, default="titanic", help="Path to data directory")
    parser.add_argument("--out", type=str, default="submission.csv", help="Submission output file")
    args = parser.parse_args()

    X_train, y_train, X_test, pids = load_and_preprocess_data(args.data_dir)
    print(f"Preprocessed features: {X_train.shape[1]} columns. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    best_model, scaler = benchmark_models(X_train, y_train)
    generate_submission(best_model, scaler, X_train, y_train, X_test, pids, args.out)
