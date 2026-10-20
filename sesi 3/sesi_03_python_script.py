import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium

def run_spatial_analysis():
    print("=== [SESI 03] Spatial Asset Proximity Analysis ===")
    
    # Simulasi Koordinat Aset Kota Bengkulu
    data = {
        'Asset_ID': ['TREE-001', 'TREE-002', 'LAMP-101', 'PARK-01', 'BUILD-05'],
        'Type': ['Pohon', 'Pohon', 'Lampu Jalan', 'Taman', 'Gedung Publik'],
        'Latitude': [-3.792, -3.795, -3.791, -3.793, -3.796],
        'Longitude': [102.260, 102.262, 102.261, 102.264, 102.263]
    }
    
    df = pd.DataFrame(data)
    geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print("[+] GeoDataFrame Aset Berhasil Dibuat:")
    print(gdf[['Asset_ID', 'Type', 'Latitude', 'Longitude']])
    
    # Render Peta Interaktif
    m = folium.Map(location=[-3.793, 102.262], zoom_start=15)
    
    for _, row in gdf.iterrows():
        color = 'green' if row['Type'] == 'Pohon' else 'blue' if row['Type'] == 'Taman' else 'red'
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"ID: {row['Asset_ID']} ({row['Type']})",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
        
        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=50,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)
        
    m.save("city_assets_map.html")
    print("\n[+] Peta Interaktif Spasial Berhasil Disimpan: 'city_assets_map.html'")

if __name__ == '__main__':
    run_spatial_analysis()