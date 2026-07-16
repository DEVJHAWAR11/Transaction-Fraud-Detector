import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def build_pipeline():
    # only Amount and Time need scaling, V1-V28 are already PCA'd
    scaler = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), ['Amount', 'Time'])
        ],
        remainder='passthrough'  # V1-V28 pass through untouched
    )
    pipeline = Pipeline(steps=[('preprocessor', scaler)])
    return pipeline

def preprocess(df, pipeline_path="models/pipeline.joblib"):
    df = df.copy()

    X = df.drop('Class', axis=1)
    y = df['Class']

    # stratified split keeps fraud ratio same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    # fit only on train, transform both — this is the key reason we use a pipeline
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # get column names back after ColumnTransformer reorders things
    scaled_cols = ['Amount', 'Time']
    passthrough_cols = [c for c in X_train.columns if c not in scaled_cols]
    final_cols = scaled_cols + passthrough_cols

    X_train_df = pd.DataFrame(X_train_processed, columns=final_cols)
    X_test_df = pd.DataFrame(X_test_processed, columns=final_cols)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # save pipeline so exact same scaling applies to future data
    joblib.dump(pipeline, pipeline_path)

    X_train_df.to_csv("data/processed/X_train.csv", index=False)
    X_test_df.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"train size: {X_train_df.shape}, test size: {X_test_df.shape}")
    print(f"fraud in train: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"fraud in test:  {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    print(f"pipeline saved to {pipeline_path}")

    return X_train_df, X_test_df, y_train, y_test


if __name__ == "__main__":
    from validate import validate
    df = pd.read_csv("data/raw/creditcard.csv")
    validate(df)
    preprocess(df)
