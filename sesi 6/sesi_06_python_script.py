import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def simulate_multispectral_radar_data(n_samples=1200):
    np.random.seed(42)
    
    vv = np.random.uniform(-20.0, -5.0, n_samples)
    vh = np.random.uniform(-25.0, -10.0, n_samples)
    red = np.random.uniform(0.02, 0.3, n_samples)
    nir = np.random.uniform(0.1, 0.6, n_samples)
    swir = np.random.uniform(0.05, 0.4, n_samples)
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    # 0: Agriculture, 1: Logging, 2: Mining, 3: Wildfire
    drivers = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.25, 0.15, 0.20])
    
    df = pd.DataFrame({
        'S1_VV': vv,
        'S1_VH': vh,
        'S2_Red': red,
        'S2_NIR': nir,
        'S2_SWIR': swir,
        'S2_NDVI': ndvi,
        'DriverLabel': drivers
    })
    return df

def run_geoai_pipeline():
    print("=== [SESI 06] GeoAI Deforestation Alert Driver Classifier ===")
    df = simulate_multispectral_radar_data()
    
    driver_names = {0: 'Agriculture', 1: 'Logging', 2: 'Mining', 3: 'Wildfire'}
    print("[+] Dataset Satelit Sentinel-1 & Sentinel-2 Berhasil Disimulasikan.")
    
    X = df.drop('DriverLabel', axis=1)
    y = df['DriverLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("\n--- Laporan Evaluasi Klasifikasi Penyebab Deforestasi ---")
    print(classification_report(y_test, y_pred, target_names=driver_names.values()))

if __name__ == '__main__':
    run_geoai_pipeline()