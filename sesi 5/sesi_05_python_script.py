import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Executive KPI Dashboard", layout="wide")

st.title("📊 Dashboard Eksekutif Penjualan (HCI Compliant)")
st.caption("Prinsip Desain: High Data-Ink Ratio | Low Cognitive Load | Direct Filtering")

@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=180)
    regions = ['Sumatera', 'Jawa', 'Kalimantan', 'Sulawesi']
    
    data = []
    for d in dates:
        for r in regions:
            sales = np.random.randint(50, 500)
            data.append({'Tanggal': d, 'Wilayah': r, 'Penjualan': sales})
    return pd.DataFrame(data)

df = load_data()

st.sidebar.header("Filter Navigasi")
selected_region = st.sidebar.multiselect("Pilih Wilayah:", df['Wilayah'].unique(), default=df['Wilayah'].unique())

filtered_df = df[df['Wilayah'].isin(selected_region)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Penjualan", f"Rp {filtered_df['Penjualan'].sum():,} Juta")
col2.metric("Rata-rata Harian", f"Rp {filtered_df['Penjualan'].mean():.1f} Juta")
col3.metric("Total Transaksi", len(filtered_df))

st.markdown("---")

st.subheader("Rata-rata Penjualan per Wilayah")
chart_data = filtered_df.groupby('Wilayah')['Penjualan'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.bar(chart_data['Wilayah'], chart_data['Penjualan'], color='#1f77b4')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel("Penjualan (Juta Rp)")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)

st.pyplot(fig)

with st.expander("🔍 Klik untuk melihat Detail Data Mentah (Details on Demand)"):
    st.dataframe(filtered_df)