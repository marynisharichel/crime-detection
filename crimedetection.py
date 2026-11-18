import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import webbrowser

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==================================================
# 1. FILE PATHS - EXACT DATASET NAME
# ==================================================

BASE_DIR = Path(__file__).parent

# Leave this as None. Only set it if you want to force one specific file, e.g.
MANUAL_CSV_PATH = r"C:\Users\maryn\Desktop\crime detetion\crime_dataset_india (1).csv"


def find_dataset():
    # 1. manual path, only if it really exists
    if MANUAL_CSV_PATH and Path(MANUAL_CSV_PATH).is_file():
        return Path(MANUAL_CSV_PATH)
    if MANUAL_CSV_PATH:
        print("MANUAL_CSV_PATH does not exist, searching instead:", MANUAL_CSV_PATH)

    # 2. search likely folders for any CSV with "crime" in the name
    search_dirs = [
        BASE_DIR,
        Path.cwd(),
        Path.home() / "Desktop",
        Path.home() / "Downloads",
        Path.home() / "Documents",
    ]
    for folder in search_dirs:
        if not folder.exists():
            continue
        found = [p for p in folder.rglob("*.csv") if "crime" in p.name.lower()]
        if found:
            return sorted(found)[0]

    # 3. last resort: ask with a file picker
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Select the crime dataset CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        root.destroy()
        if chosen:
            return Path(chosen)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find the crime dataset CSV.\n"
        f"Script folder: {BASE_DIR}\n"
        "Copy the CSV into that folder and run again."
    )

file_path = find_dataset()
print("Using dataset:", file_path)

output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)

# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(file_path, encoding='utf-8-sig')

df.columns = df.columns.str.strip()

print("\n========== DATASET INFORMATION ==========")
print("Dataset:", file_path.name)
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())

# ==================================================
# 3. DATA PREPROCESSING
# ==================================================

df['Date of Occurrence'] = pd.to_datetime(
    df['Date of Occurrence'],
    format='%d-%m-%Y %H:%M',
    errors='coerce'
)

df['City'] = df['City'].astype(str).str.strip().str.title()

df = df.dropna(
    subset=['Date of Occurrence', 'City']
)

df = df[df['City'] != 'Nan']

df['Year'] = df['Date of Occurrence'].dt.year
df['Month'] = df['Date of Occurrence'].dt.month
df['Day'] = df['Date of Occurrence'].dt.day

print("\nRecords after cleaning:", len(df))
print("Number of cities:", df['City'].nunique())

# ==================================================
# 4. CITY-WISE CRIME COUNT
# ==================================================

city_crime = (
    df.groupby('City')
    .size()
    .sort_values(ascending=False)
)

top_unsafe = city_crime.head(5)

top_safe = city_crime.tail(5).sort_values()

print("\n========== TOP 5 HIGHEST CRIME COUNT CITIES ==========")
print(top_unsafe)

print("\n========== TOP 5 LOWEST CRIME COUNT CITIES ==========")
print(top_safe)

# ==================================================
# 5. GRAPH - TOP 5 HIGHEST CRIME COUNT CITIES
# ==================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_unsafe.index,
    y=top_unsafe.values,
    color='red'
)

plt.title("Top 5 Cities with Highest Recorded Crime Counts")
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(output_dir / "top_5_unsafe_cities.png")
plt.close()

# ==================================================
# 6. GRAPH - TOP 5 LOWEST CRIME COUNT CITIES
# ==================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_safe.index,
    y=top_safe.values,
    color='green'
)

plt.title("Top 5 Cities with Lowest Recorded Crime Counts")
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(output_dir / "top_5_lowest_crime_cities.png")
plt.close()

# ==================================================
# 7. CITY-WISE CRIME GRAPH
# ==================================================

plt.figure(figsize=(14, 7))

city_crime.plot(
    kind='bar',
    color='orange'
)

plt.title("City-Wise Crime Analysis")
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "city_wise_crime.png")
plt.close()

# ==================================================
# 8. CRIME DOMAIN GRAPH
# ==================================================

domain_counts = df['Crime Domain'].value_counts()

plt.figure(figsize=(9, 6))

domain_counts.plot(
    kind='bar',
    color='purple'
)

plt.title("Crime Domain Analysis")
plt.xlabel("Crime Domain")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=25)

plt.tight_layout()
plt.savefig(output_dir / "crime_domain.png")
plt.close()

# ==================================================
# 9. CRIME DESCRIPTION GRAPH
# ==================================================

crime_types = df['Crime Description'].value_counts().head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=crime_types.values,
    y=crime_types.index,
    color='teal'
)

plt.title("Top 10 Most Reported Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Description")

plt.tight_layout()
plt.savefig(output_dir / "crime_types.png")
plt.close()

# ==================================================
# 10. MONTHLY CRIME TREND
# ==================================================

monthly_crime = (
    df.groupby(
        df['Date of Occurrence'].dt.to_period('M')
    ).size()
)

plt.figure(figsize=(13, 6))

plt.plot(
    monthly_crime.index.astype(str),
    monthly_crime.values,
    marker='o',
    color='blue'
)

plt.title("Monthly Crime Trend")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=60)
plt.grid(True)

plt.tight_layout()
plt.savefig(output_dir / "monthly_crime_trend.png")
plt.close()

# ==================================================
# 11. HOTSPOT DETECTION
# ==================================================

grp = (
    df.groupby(['City', df['Date of Occurrence'].dt.date])
    .size()
    .reset_index(name='count')
)

grp.rename(
    columns={'Date of Occurrence': 'Date'},
    inplace=True
)

grp['Date'] = pd.to_datetime(grp['Date'])

grp['month'] = grp['Date'].dt.month
grp['day'] = grp['Date'].dt.day
grp['year'] = grp['Date'].dt.year

# Hotspot: 2 or more recorded crimes per city per day

grp['hotspot'] = (
    grp['count'] >= 2
).astype(int)

print("\n========== HOTSPOT ANALYSIS ==========")
print("Total city-date records:", len(grp))
print("Hotspot records:", grp['hotspot'].sum())

# ==================================================
# 12. RANDOM FOREST CLASSIFICATION
# ==================================================

features = ['count', 'month', 'day', 'year']

X = grp[features]
y = grp['hotspot']

if y.nunique() == 2 and y.value_counts().min() >= 2:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== RANDOM FOREST RESULTS ==========")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

else:
    print("\nRandom Forest skipped.")
    print("Insufficient class variation.")

# ==================================================
# 13. CITY COORDINATES
# ==================================================

city_coords = {

    'Delhi': [28.6139, 77.2090],
    'Mumbai': [19.0760, 72.8777],
    'Bangalore': [12.9716, 77.5946],
    'Hyderabad': [17.3850, 78.4867],
    'Kolkata': [22.5726, 88.3639],
    'Chennai': [13.0827, 80.2707],
    'Pune': [18.5204, 73.8567],
    'Ahmedabad': [23.0225, 72.5714],
    'Jaipur': [26.9124, 75.7873],
    'Lucknow': [26.8467, 80.9462],
    'Kanpur': [26.4499, 80.3319],
    'Surat': [21.1702, 72.8311],
    'Nagpur': [21.1458, 79.0882],
    'Agra': [27.1767, 78.0081],
    'Ludhiana': [30.9010, 75.8573],
    'Visakhapatnam': [17.6868, 83.2185],
    'Thane': [19.2183, 72.9781],
    'Ghaziabad': [28.6692, 77.4538],
    'Indore': [22.7196, 75.8577],
    'Patna': [25.5941, 85.1376],
    'Bhopal': [23.2599, 77.4126],
    'Meerut': [28.9845, 77.7064],
    'Srinagar': [34.0837, 74.7973],
    'Nashik': [19.9975, 73.7898],
    'Vasai': [19.3919, 72.8397],
    'Varanasi': [25.3176, 82.9739],
    'Kalyan': [19.2437, 73.1355],
    'Faridabad': [28.4089, 77.3178],
    'Rajkot': [22.3039, 70.8022]

}

# ==================================================
# 14. CREATE INDIA CRIME MAP
# ==================================================

# Free, key-free basemaps that load fine from a local HTML file.
# (OpenStreetMap's own server returns 403 for file:// pages, and CARTO now
# requires an API key.)
crime_map = folium.Map(
    location=[22.5, 79.0],
    zoom_start=5,
    tiles=None
)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles &copy; Esri',
    name='Street map',
    max_zoom=18
).add_to(crime_map)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles &copy; Esri',
    name='Light gray map',
    max_zoom=16
).add_to(crime_map)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles &copy; Esri, Maxar, Earthstar Geographics',
    name='Satellite',
    max_zoom=18
).add_to(crime_map)

mapped_cities = 0

for city, count in city_crime.items():

    if city not in city_coords:
        continue

    if city in top_unsafe.index:
        color = 'red'
        status = 'Highest recorded crime count'

    elif city in top_safe.index:
        color = 'green'
        status = 'Lowest recorded crime count'

    else:
        color = 'orange'
        status = 'Other city'

    lat, lon = city_coords[city]

    popup = f"""
    <b>City:</b> {city}<br>
    <b>Total Crimes:</b> {count}<br>
    <b>Status:</b> {status}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=9,
        popup=folium.Popup(popup, max_width=250),
        tooltip=f"{city} - {count} crimes",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8
    ).add_to(crime_map)

    mapped_cities += 1

folium.LayerControl(collapsed=True).add_to(crime_map)

# ==================================================
# 15. MAP LEGEND
# ==================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 240px;
background: white;
border: 2px solid grey;
z-index: 9999;
padding: 12px;
font-size: 14px;
">

<b>CRIME HOTSPOT MAP</b><br><br>

<span style="color:red;">●</span>
Highest recorded crime counts<br>

<span style="color:green;">●</span>
Lowest recorded crime counts<br>

<span style="color:orange;">●</span>
Other cities<br>

</div>
"""

crime_map.get_root().html.add_child(
    folium.Element(legend_html)
)

# ==================================================
# 16. SAVE AND OPEN MAP
# ==================================================

map_path = output_dir / "crime_hotspot_map.html"

crime_map.save(str(map_path))

print("\n========== MAP INFORMATION ==========")

print("Map created successfully!")
print("Total mapped cities:", mapped_cities)
print("Map location:", map_path)

import os, sys, subprocess
map_abs = map_path.resolve()
print("Map file:", map_abs)
opened = False
try:
    opened = webbrowser.open(map_abs.as_uri())
except Exception:
    opened = False
if not opened:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(map_abs))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(map_abs)])
        else:
            subprocess.run(["xdg-open", str(map_abs)])
    except Exception:
        print("Could not auto-open. Double-click this file to view the map:", map_abs)

# ==================================================
# 17. FINAL SUMMARY
# ==================================================

print("\n========== PROJECT SUMMARY ==========")

print("Dataset records:", len(df))
print("Total cities:", df['City'].nunique())

print("\nHighest recorded crime count:")
print(city_crime.idxmax(), city_crime.max())

print("\nLowest recorded crime count:")
print(city_crime.idxmin(), city_crime.min())

print("\nAll graphs and map saved in output folder.")

print("\nCRIME HOTSPOT DETECTION COMPLETED!")