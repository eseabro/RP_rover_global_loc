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

# def build_catalog(n=1000, seed=0, region=None, elevation_range=(-4000, 8000)):
#     """
#     Generate a synthetic terrain feature catalog for a Mars rover.
    
#     Parameters
#     ----------
#     n : int
#         Number of terrain features to generate.
#     seed : int
#         Random seed for reproducibility.
#     region : tuple or None
#         Optional (lat_min, lat_max, lon_min, lon_max) defining area of interest.
#         If None, features are distributed globally.
#     elevation_range : tuple
#         Range of elevation values in meters.
        
#     Returns
#     -------
#     catalog : list of dicts
#         Each entry has: id, lat_deg, lon_deg, elev_m, reflectance, vec
#     """
#     rng = np.random.RandomState(seed)

#     if region is None:
#         lats = rng.uniform(-90, 90, size=n)
#         lons = rng.uniform(-180, 180, size=n)
#     else:
#         lat_min, lat_max, lon_min, lon_max = region
#         lats = rng.uniform(lat_min, lat_max, size=n)
#         lons = rng.uniform(lon_min, lon_max, size=n)

#     reflectance = np.round(rng.uniform(0.1, 1.0, size=n), 3)  # simulates surface brightness
#     vecs = lat_lon_to_vector(lats, lons)

#     catalog = [{
#         "id": int(i + 1),
#         "lat_deg": float(lats[i]),
#         "lon_deg": float(lons[i]),
#         "reflectance": float(reflectance[i]),
#         "vec": vecs[i]
#     } for i in range(n)]

#     return catalog

def build_catalog(n=1000, seed=0, region=None, mars_radius_km=3390.0):
    rng = np.random.RandomState(seed)

    if region is None:
        lats = rng.uniform(-90, 90, size=n)
        lons = rng.uniform(-180, 180, size=n)
    else:
        lat_min, lat_max, lon_min, lon_max = region
        lats = rng.uniform(lat_min, lat_max, size=n)
        lons = rng.uniform(lon_min, lon_max, size=n)

    reflectance = np.round(rng.uniform(0.1, 1.0, size=n), 3)
    vecs = lat_lon_to_vector(lats, lons)

    # Reference point for tangent plane (center of region)
    ref_lat = np.mean(lats)
    ref_lon = np.mean(lons)
    ref_vec = lat_lon_to_vector(np.array([ref_lat]), np.array([ref_lon]))[0]
    up = ref_vec / np.linalg.norm(ref_vec)
    east = np.cross([0,0,1], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)

    catalog = []
    for i in range(n):
        v = vecs[i] - ref_vec
        x = float(np.dot(v, east))
        y = float(np.dot(v, north))
        catalog.append({
            "id": int(i + 1),
            "lat_deg": float(lats[i]),
            "lon_deg": float(lons[i]),
            "reflectance": float(reflectance[i]),
            "x": x,
            "y": y
        })
    return catalog


def build_mars_catalog(n=1000, region=None, seed=0, mars_radius_km=3390.0, elev_range_km=(-2, 5)):
    """
    Build a synthetic Mars landmark catalog with lat/lon/x/y/elev coordinates.
    (x=0, y=0, z=0 corresponds to lat=0°, lon=0°.)

    Args:
        n: number of landmarks
        region: (lat_min, lat_max, lon_min, lon_max) or None for global
        seed: random seed
        mars_radius_km: mean Mars radius
        elev_range_km: (min, max) elevation variation
    """
    rng = np.random.RandomState(seed)

    # --- Random lat/lon within region or globally ---
    if region is None:
        lats = rng.uniform(-90, 90, size=n)
        lons = rng.uniform(-180, 180, size=n)
    else:
        lat_min, lat_max, lon_min, lon_max = region
        lats = rng.uniform(lat_min, lat_max, size=n)
        lons = rng.uniform(lon_min, lon_max, size=n)

    # --- Random elevation offsets ---
    elev = rng.uniform(elev_range_km[0], elev_range_km[1], size=n)

    # --- Convert to radians ---
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)

    # --- Convert to Cartesian coordinates ---
    r = mars_radius_km + elev
    # By convention: x-axis → lon=0°, lat=0° ; y-axis → 90°E ; z-axis → north pole
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    # --- Optional: reflectance or brightness ---
    reflectance = np.round(rng.uniform(0.1, 1.0, size=n), 3)

    # --- Assemble catalog ---
    catalog = [{
        "id": int(i + 1),
        "lat_deg": float(lats[i]),
        "lon_deg": float(lons[i]),
        "x": float(x[i]),
        "y": float(y[i]),
        "elev": float(elev[i]),
        "reflectance": float(reflectance[i])
    } for i in range(n)]

    return catalog

def save_catalog_csv(catalog, path):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lat_deg", "lon_deg", "reflectance", "vec_x", "vec_y", "vec_z"])
        for c in catalog:
            writer.writerow([
                c["id"], c["lat_deg"], c["lon_deg"], 
                c["reflectance"], c["x"], c["y"]
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
