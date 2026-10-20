import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

def supervised_house_price():
    print("\n--- 1. Supervised Learning: Prediksi Harga Rumah ---")
    np.random.seed(42)
    size_sqm = np.random.randint(36, 300, 200)
    bedrooms = np.random.randint(1, 5, 200)
    price = size_sqm * 12_000_000 + bedrooms * 25_000_000 + np.random.normal(0, 50_000_000, 200)
    
    X = pd.DataFrame({'SizeSqm': size_sqm, 'Bedrooms': bedrooms})
    y = price
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"[+] RMSE: IDR {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"[+] R2 Score: {r2_score(y_test, y_pred):.4f}")

def unsupervised_customer_segmentation():
    print("\n--- 2. Unsupervised Learning: Segmentasi K-Means ---")
    np.random.seed(42)
    annual_spend = np.random.randint(10, 500, 300)
    visit_frequency = np.random.randint(1, 50, 300)
    
    X = np.column_stack((annual_spend, visit_frequency))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print(f"[+] K-Means Selesai (K=3). Distribusi Cluster:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    - Cluster {u}: {c} Pelanggan")

if __name__ == '__main__':
    print("=== [SESI 04] Machine Learning End-to-End Workflow ===")
    supervised_house_price()
    unsupervised_customer_segmentation()