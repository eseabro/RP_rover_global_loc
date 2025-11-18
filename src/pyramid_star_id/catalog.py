"""Catalog utilities: build or load small catalogs (RA/Dec -> unit vectors)"""
import numpy as np
import csv

def lat_lon_to_vector(lat_deg, lon_deg):
    """Convert planet-fixed latitude/longitude (degrees) to 3D unit vector."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    v = np.stack([x, y, z], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def build_catalog(n=1000, seed=0, region=None):
    rng = np.random.RandomState(seed)

    if region is None:
        lats = rng.uniform(-90, 90, size=n)
        lons = rng.uniform(-180, 180, size=n)
    else:
        lat_min, lat_max, lon_min, lon_max = region
        lats = rng.uniform(lat_min, lat_max, size=n)
        lons = rng.uniform(lon_min, lon_max, size=n)

    reflectance = np.round(rng.uniform(0.1, 1.0, size=n), 3)

    catalog = []
    for i in range(n):
        catalog.append({
            "id": int(i + 1),
            "lat_deg": float(lats[i]),
            "lon_deg": float(lons[i]),
            "reflectance": float(reflectance[i])
        })
    print('Saving catalog...')
    save_catalog_csv(catalog, f'output/catalog_{n}_{region[1]-region[0]}_{region[3]-region[2]}.csv')

    return catalog


def save_catalog_csv(catalog, path):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lat_deg", "lon_deg", "reflectance", "x", "y", "elev"])
        for c in catalog:
            writer.writerow([
                c["id"], c["lat_deg"], c["lon_deg"], 
                c["reflectance"]
            ])


def load_catalog(path):
    catalog = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalog.append({
                "id": int(row["id"]),
                "lat_deg": float(row["lat_deg"]),
                "lon_deg": float(row["lon_deg"]),
                "reflectance": float(row.get("reflectance", 9.9)),
                "vec": np.array([float(row["vec_x"]), float(row["vec_y"]), float(row["vec_z"])])
            })
    return catalog
