import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path):
    df = pd.read_csv(train_path)
    
    X = df[['id']]
    y = df[['jenis', 'warna']]
    
    return X, y

def train_multilabel_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model = MultiOutputClassifier(base_classifier)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    target_names = ['jenis', 'warna']
    for idx, target in enumerate(target_names):
        print(f"\nClassification Report for {target}:")
        print(classification_report(
            y_test.iloc[:, idx],
            y_pred[:, idx]
        ))

def predict_new_data(model, scaler, test_path):
    test_df = pd.read_csv(test_path)
    X_test = test_df[['id']]
    
    X_test_scaled = scaler.transform(X_test)
    
    predictions = model.predict(X_test_scaled)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'jenis': predictions[:, 0],
        'warna': predictions[:, 1]
    })
    
    return submission

if __name__ == "__main__":
    X, y = load_data('/home/remote/Hology-7-Data-Mining-Competition-2024/Penyisihan Hology Data Mining/train.csv')
    
    print("Training model...")
    model, scaler, X_test, y_test = train_multilabel_model(X, y)
    
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("\nMaking predictions on test set...")
    submission = predict_new_data(model, scaler, '/home/remote/Hology-7-Data-Mining-Competition-2024/Penyisihan Hology Data Mining/sample_submission.csv')
    
    submission.to_csv('/home/remote/Hology-7-Data-Mining-Competition-2024/prediction/pred-base-1.csv', index=False)
    print("\nPredictions saved to 'predictions.csv'")