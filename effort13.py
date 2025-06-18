#!/usr/bin/env python3
"""
effort13.py — Interactive Injury Recurrence Predictor (With Height & Weight)
============================================================================
This script trains a neural-network classifier to predict injury recurrence,
including Height and Weight as additional numeric features. It provides
detailed prompts indicating what each numeric code represents.

Encoded features and their representations:
  - Age_code: 1 if Age < 25, else 0
  - Gender_code: 1 = Male, 0 = Female
  - Height: numeric in centimeters
  - Weight: numeric in kilograms
  - BMI_code: 1 if BMI < 30, else 0
  - Activity_code: 0 = Sedentary, 1 = Recreational, 2 = Competitive
  - Muscle_code: 0 = Bone patellar tendon bone graft, 1 = Hamstrings, 2 = Peroneus longus
    (values in sheet already numeric-coded as 0/1/2)
  - DiamFem_code: 1 if Diameter of graft (Femur) < 8 mm, else 0
  - DiamTib_code: 1 if Diameter of graft (tibia) < 8 mm, else 0
  - FemFix_code: 0 = Direct compression type device, 1 = Expansion type device, 2 = Suspension type device
    (values in sheet already numeric-coded as 0/1/2)
  - Rehab_code: 0 = Accelerated, 1 = Conventional
    (values in sheet already numeric-coded as 0 or 1)
  - Beighton_code: 1 if Beighton's score >= 5, else 0
  - MMTQuad_code: 1 if MMT Quadriceps < 3, else 0
  - MMTHam_code: 1 if MMT Hamstrings < 3, else 0

Usage:
  Train:
      python3 effort13.py train data.xlsx model.pkl
  Chat:
      python3 effort13.py chat model.pkl
"""

import argparse
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path: Path) -> pd.DataFrame:
    """Load Excel/CSV and strip column names."""
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def encode_dataframe(df: pd.DataFrame) -> (pd.DataFrame, list):
    """Encode raw DataFrame into numeric features, including Height and Weight."""
    data = df.copy()
    feature_cols = []

    # Age: <25 => 1 else 0
    if 'Age' in data.columns:
        data['Age_num'] = pd.to_numeric(data['Age'], errors='coerce')
        data['Age_code'] = data['Age_num'].apply(lambda x: 1 if pd.notnull(x) and x < 25 else 0)
        feature_cols.append('Age_code')

    # Gender: 1 = Male, 0 = Female
    if 'Gender' in data.columns:
        data['Gender_code'] = data['Gender'].astype(str).str.strip().map({'1.0': 1, '0.0': 0, '1':1, '0':0}).fillna(0).astype(int)
        feature_cols.append('Gender_code')

    # Height in centimeters (numeric feature)
    if 'Height' in data.columns:
        data['Height_num'] = pd.to_numeric(data['Height'], errors='coerce')
        feature_cols.append('Height_num')

    # Weight in kilograms (numeric feature)
    if 'Weight' in data.columns:
        data['Weight_num'] = pd.to_numeric(data['Weight'], errors='coerce')
        feature_cols.append('Weight_num')

    # BMI: <30 => 1 else 0
    if 'BMI' in data.columns:
        data['BMI_num'] = pd.to_numeric(data['BMI'], errors='coerce')
        data['BMI_code'] = data['BMI_num'].apply(lambda x: 1 if pd.notnull(x) and x < 30 else 0)
        feature_cols.append('BMI_code')

    # Activity level: 0 = Sedentary, 1 = Recreational, 2 = Competitive
    if 'Activity level' in data.columns:
        data['Activity_code'] = pd.to_numeric(data['Activity level'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('Activity_code')

    # Muscles used for Graft: 0 = Bone patellar tendon bone graft, 1 = Hamstrings, 2 = Peroneus longus
    if 'Muscles used for Graft' in data.columns:
        data['Muscle_code'] = pd.to_numeric(data['Muscles used for Graft'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('Muscle_code')

    # Diameter of graft (Femur): <8 => 1 else 0
    if 'Diameter of graft(Femur)' in data.columns:
        data['DiamFem_num'] = pd.to_numeric(data['Diameter of graft(Femur)'], errors='coerce')
        data['DiamFem_code'] = data['DiamFem_num'].apply(lambda x: 1 if pd.notnull(x) and x < 8 else 0)
        feature_cols.append('DiamFem_code')

    # Diameter of graft (tibia): <8 => 1 else 0
    if 'Diameter of graft(tibia)' in data.columns:
        data['DiamTib_num'] = pd.to_numeric(data['Diameter of graft(tibia)'], errors='coerce')
        data['DiamTib_code'] = data['DiamTib_num'].apply(lambda x: 1 if pd.notnull(x) and x < 8 else 0)
        feature_cols.append('DiamTib_code')

    # Femoral fixation: 0 = Direct compression type device, 1 = Expansion type device, 2 = Suspension type device
    if 'Femoral fixation' in data.columns:
        data['FemFix_code'] = pd.to_numeric(data['Femoral fixation'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('FemFix_code')

    # Rehabilitation Protocol: 0 = Accelerated, 1 = Conventional
    if 'Rehabilitation Protocol' in data.columns:
        data['Rehab_code'] = pd.to_numeric(data['Rehabilitation Protocol'], errors='coerce').fillna(0).astype(int)
        feature_cols.append('Rehab_code')

    # Beighton's score: >=5 => 1 else 0
    if "Beighton's score" in data.columns:
        data['Beighton_num'] = pd.to_numeric(data["Beighton's score"], errors='coerce')
        data['Beighton_code'] = data['Beighton_num'].apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)
        feature_cols.append('Beighton_code')

    # MMT Quadriceps: <3 => 1 else 0
    if 'MMT Quadriceps' in data.columns:
        data['MMTQuad_num'] = pd.to_numeric(data['MMT Quadriceps'], errors='coerce')
        data['MMTQuad_code'] = data['MMTQuad_num'].apply(lambda x: 1 if pd.notnull(x) and x < 3 else 0)
        feature_cols.append('MMTQuad_code')

    # MMT Hamstrings: <3 => 1 else 0
    if 'MMT Hamstrings' in data.columns:
        data['MMTHam_num'] = pd.to_numeric(data['MMT Hamstrings'], errors='coerce')
        data['MMTHam_code'] = data['MMTHam_num'].apply(lambda x: 1 if pd.notnull(x) and x < 3 else 0)
        feature_cols.append('MMTHam_code')

    if not feature_cols:
        raise ValueError("No supported feature columns found in the data.")

    return data[feature_cols], feature_cols


def preprocess(df: pd.DataFrame):
    """Encode features, drop NaN in Recurrence, and build preprocessing pipeline."""
    if 'Recurrence' not in df.columns:
        raise KeyError("Target column 'Recurrence' not found. Ensure it exists.")

    df = df.dropna(subset=['Recurrence'])
    X_raw, feature_names = encode_dataframe(df)
    y = pd.to_numeric(df['Recurrence'], errors='coerce').astype(int)

    transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return X_raw, y, transformer, feature_names


def build_model():
    """Return an MLPClassifier (neural network)."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=500, random_state=42)


def train(data_path: Path, model_path: Path):
    df = load_data(data_path)
    X_raw, y, transformer, feature_names = preprocess(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(steps=[
        ('prep', transformer),
        ('clf', build_model())
    ])

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Validation Results:")
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred, digits=3))

    joblib.dump({'model': pipeline, 'features': feature_names}, model_path)
    print(f"[INFO] Model saved to {model_path}")


def chat(model_path: Path):
    """Interactive prompt: enter raw values, encode, and predict probability."""
    data = joblib.load(model_path)
    pipeline = data['model']
    feature_names = data['features']

    print("Entering chat mode. Enter raw values (type 'exit' to quit):\n")
    while True:
        inputs = {}
        # Age
        if 'Age_code' in feature_names:
            val = input("Age (years): ")
            if val.strip().lower() == 'exit': break
            try: inputs['Age'] = float(val)
            except: inputs['Age'] = np.nan
        # Gender
        if 'Gender_code' in feature_names:
            val = input("Gender (1 = Male, 0 = Female): ").strip()
            if val.lower() == 'exit': break
            try: inputs['Gender'] = float(val)
            except: inputs['Gender'] = 0
        # Height
        if 'Height_num' in feature_names:
            val = input("Height (cm): ")
            try: inputs['Height'] = float(val)
            except: inputs['Height'] = np.nan
        # Weight
        if 'Weight_num' in feature_names:
            val = input("Weight (kg): ")
            try: inputs['Weight'] = float(val)
            except: inputs['Weight'] = np.nan
        # BMI
        if 'BMI_code' in feature_names:
            val = input("BMI: ")
            try: inputs['BMI'] = float(val)
            except: inputs['BMI'] = np.nan
        # Activity level
        if 'Activity_code' in feature_names:
            val = input("Activity level (0 = Sedentary, 1 = Recreational, 2 = Competitive): ").strip()
            if val.lower() == 'exit': break
            try: inputs['Activity level'] = float(val)
            except: inputs['Activity level'] = 0
        # Muscles used for Graft
        if 'Muscle_code' in feature_names:
            val = input("Muscles used for Graft (0 = Bone patellar tendon bone graft, 1 = Hamstrings, 2 = Peroneus longus): ").strip()
            if val.lower() == 'exit': break
            try: inputs['Muscles used for Graft'] = float(val)
            except: inputs['Muscles used for Graft'] = 0
        # Diameter of graft (Femur)
        if 'DiamFem_code' in feature_names:
            val = input("Diameter of graft (Femur) in mm: ")
            try: inputs['Diameter of graft(Femur)'] = float(val)
            except: inputs['Diameter of graft(Femur)'] = np.nan
        # Diameter of graft (tibia)
        if 'DiamTib_code' in feature_names:
            val = input("Diameter of graft (tibia) in mm: ")
            try: inputs['Diameter of graft(tibia)'] = float(val)
            except: inputs['Diameter of graft(tibia)'] = np.nan
        # Femoral fixation
        if 'FemFix_code' in feature_names:
            val = input("Femoral fixation (0 = Direct compression type device, 1 = Expansion type device, 2 = Suspension type device): ").strip()
            if val.lower() == 'exit': break
            try: inputs['Femoral fixation'] = float(val)
            except: inputs['Femoral fixation'] = 0
        # Rehabilitation Protocol
        if 'Rehab_code' in feature_names:
            val = input("Rehabilitation Protocol (0 = Accelerated, 1 = Conventional): ").strip()
            if val.lower() == 'exit': break
            try: inputs['Rehabilitation Protocol'] = float(val)
            except: inputs['Rehabilitation Protocol'] = 0
        # Beighton's score
        if 'Beighton_code' in feature_names:
            val = input("Beighton's score (0-9): ")
            try: inputs["Beighton's score"] = float(val)
            except: inputs["Beighton's score"] = np.nan
        # MMT Quadriceps
        if 'MMTQuad_code' in feature_names:
            val = input("MMT Quadriceps (0-5): ")
            try: inputs['MMT Quadriceps'] = float(val)
            except: inputs['MMT Quadriceps'] = np.nan
        # MMT Hamstrings
        if 'MMTHam_code' in feature_names:
            val = input("MMT Hamstrings (0-5): ")
            try: inputs['MMT Hamstrings'] = float(val)
            except: inputs['MMT Hamstrings'] = np.nan

        df_row = pd.DataFrame([inputs])
        X_row, _ = encode_dataframe(df_row)
        proba = pipeline.predict_proba(X_row)[0][1]
        print(f"\nPredicted probability of recurrence: {proba:.4f}\n")

    print("Exiting chat mode.")


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict injury recurrence with detailed prompts."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train', help="Train a new model")
    train_p.add_argument('data', type=Path, help="Training data (xlsx/csv)")
    train_p.add_argument('model', type=Path, help="Path to save model (e.g., model.pkl)")

    chat_p = subparsers.add_parser('chat', help="Interactive chat for prediction")
    chat_p.add_argument('model', type=Path, help="Trained model file (pkl)")

    args = parser.parse_args()
    if args.command == 'train':
        train(args.data, args.model)
    elif args.command == 'chat':
        chat(args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
