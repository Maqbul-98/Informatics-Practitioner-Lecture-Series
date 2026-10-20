import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

def generate_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(18, 65, size=n_samples)
    income = np.random.randint(2000, 20000, size=n_samples)
    tenure_months = np.random.randint(1, 60, size=n_samples)
    usage_score = np.random.uniform(10.0, 100.0, size=n_samples)
    
    # Target: High-Risk (1) vs Low-Risk (0)
    target = (income < 5000) & (tenure_months < 12) | (usage_score < 30.0)
    target = target.astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'TenureMonths': tenure_months,
        'UsageScore': usage_score,
        'RiskTarget': target
    })
    return df

def run_pipeline():
    print("=== [SESI 01] Automated Data Mining & AI Pipeline ===")
    df = generate_data()
    print(f"[+] Dataset Dibuat: {df.shape[0]} Baris, {df.shape[1]} Kolom")
    
    X = df.drop('RiskTarget', axis=1)
    y = df['RiskTarget']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    
    print("\n--- Model Evaluation Report ---")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    with open('pipeline_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("[+] Laporan berhasil disimpan ke 'pipeline_report.json'")

if __name__ == '__main__':
    run_pipeline()