import csv
import numpy as np
import os
import sys

# Import your existing logic
# (Ensure your identifier.py has the fast hash builder we just wrote!)
from identifier import identify_geometric
import matplotlib.pyplot as plt
import numpy as np

def plot_match_result(global_pts, local_pts):
    """
    Plots the Global Map (Catalog) and projects the Local Map (Observed) 
    onto it using the estimated pose.

    Args:
        global_pts: (N, 2) numpy array of Global Map [X, Y]
        local_pts:  (M, 2) numpy array of Local Map [X, Y]
        match_result: dict returned from identify_geometric() containing 'best_solution'
    """
    
    # 1. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 2. Plot All Global Points (Gray Background)
    ax.scatter(global_pts[:, 0], global_pts[:, 1], 
               c='lightgray', s=50, alpha=0.6, label='Global Map (Catalog)')

    
    ax.scatter(local_pts[:, 0], local_pts[:, 1], 
                   c='orange', marker='x', s=30, label='Local Rocks')

    # 9. Formatting
    ax.set_xlabel("Global X (meters)")
    ax.set_ylabel("Global Y (meters)")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def load_map_with_sizes(filepath):
    """
    Reads Map_X, Map_Y, Width_m, Length_m from CSV.
    Returns:
        points: (N, 2) array of X, Y
        sizes:  (N, 2) array of Width, Length
        ids:    List of IDs
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None, None, None

    points = []
    sizes = []
    ids = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        for row in reader:
            try:
                # Position
                x = float(row['Map_X'])
                y = float(row['Map_Y'])
                
                # Size (Default to 0.5m if missing to prevent crashes)
                w = float(row.get('Width_m', 0.5))
                l = float(row.get('Length_m', 0.5))
                
                points.append([x, y])
                sizes.append([w, l]) # Keep width/length separate
                ids.append(row['ID'])
            except ValueError:
                continue

    return np.array(points, dtype=np.float32), np.array(sizes, dtype=np.float32), ids

def match_local_to_global(global_csv, local_csv):
    print(f"Loading Global Map: {global_csv}")
    global_pts, global_sizes, global_ids = load_map_with_sizes(global_csv)
    
    print(f"Loading Local Map:  {local_csv}")
    local_pts, local_sizes, local_ids = load_map_with_sizes(local_csv)

    if global_pts is None or local_pts is None: return

    plot_match_result(global_pts, local_pts)
    # --- CONFIGURATION ---
    # Relaxed Matching Params
    match_percentage = 0.60  # Only require 60% of local rocks to find a match (Safe starting point)
    size_tolerance = 0.50    # Allow 50% size difference (Satellite vs Ground segmentation varies a lot)
    
    print(f"Points: Global={len(global_pts)}, Local={len(local_pts)}")
    print(f"Goal: Match at least {int(len(local_pts) * match_percentage)} rocks.")

    # --- RUN GEOMETRIC HASHING ---
    # We pass the sizes effectively as 'metadata' to the identifier if we modified it,
    # but since identifier.py is generic, we will run the geometric match first
    # and then FILTER the result using sizes.
    
    # However, to use sizes INSIDE the loop for speed, we would need to edit identifier.py.
    # For now, let's use a robust verification step AFTER the geometric match.
    
    catalog_input = global_pts 
    sim_input = {
        'observed_vectors': local_pts,
        'n_false': int(len(local_pts) * (1.0 - match_percentage)) # Allow some "false" (unmatched) rocks
    }

    result = identify_geometric(
        sim_input, 
        catalog_input,
        eps=3.0,            # Increased error tolerance to 3.0 meters
        binsize=0.2,        # Increased bin size for fuzzier shape matching
        ransac_iters=10000, # More iterations to find the "needle in the haystack"
        min_seed_inliers=3,
        early_exit_fraction=match_percentage # Stop if we hit 80% (or whatever you set)
    )

    best = result['best_solution']

    if best:
        # --- SIZE VERIFICATION STEP ---
        # The geometric matcher found a shape match. Now let's check if the sizes make sense.
        # This filters out "False Positives" where the shape is right but the rocks are wrong.
        
        matches = best['matches']
        valid_matches = []
        
        print("\n--- Verifying Sizes of Matched Rocks ---")
        
        for (cat_idx, obs_idx, resid) in matches:
            # Get dimensions (Area is a robust metric)
            g_area = global_sizes[cat_idx][0] * global_sizes[cat_idx][1]
            l_area = local_sizes[obs_idx][0] * local_sizes[obs_idx][1]
            
            # Check Ratio
            ratio = l_area / (g_area + 1e-6)
            
            # If the local rock is between 0.5x and 2.0x the size of the global rock, keep it.
            if (1.0 - size_tolerance) < ratio < (1.0 + size_tolerance):
                valid_matches.append((cat_idx, obs_idx, resid))
            else:
                # Optional: Uncomment to see rejected matches
                # print(f"  Rejecting match: G-ID {global_ids[cat_idx]} vs L-ID {local_ids[obs_idx]} (Size Mismatch)")
                pass

        # Re-assess success based on VALID matches only
        success_ratio = len(valid_matches) / len(local_pts)
        
        if success_ratio >= match_percentage:
            print("\n✅ ROBUST MATCH CONFIRMED!")
            print(f"Geometric Inliers: {len(matches)}")
            print(f"Size-Verified Inliers: {len(valid_matches)} ({success_ratio*100:.1f}%)")
            
            # Recalculate pose might be needed if we dropped too many outliers, 
            # but usually the original R/t is fine if the geometric consensus was strong.
            t = best['t']
            R = best['R']
            theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            print(f"\n--- ESTIMATED POSE ---")
            print(f"Rotation:    {theta_deg:.2f} degrees")
            print(f"Translation: X={t[0]:.2f}, Y={t[1]:.2f}")
            
            print("\n--- CONFIRMED MATCHES ---")
            for (c, o, r) in valid_matches:
                 print(f"  Local Rock {local_ids[o]} -> Global Rock {global_ids[c]}")
                 
        else:
            print(f"\n⚠️ Geometric match found, but failed size verification.")
            print(f"Valid Size Matches: {len(valid_matches)}/{len(local_pts)} ({success_ratio*100:.1f}%)")
            print("Action: Try increasing 'size_tolerance' or checking your segmentation scale.")

    else:
        print("\n❌ NO MATCH FOUND.")
        print("Tips:")
        print("1. Increase 'eps' (tolerance).")
        print("2. Increase 'match_percentage' (maybe you only see 50% of the rocks?).")


if __name__ == "__main__":
    # Replace with your actual file paths
    global_csv_path = "/home/ws/src/hirise_data/above_rock_analysis.csv" # The "Above" Map
    local_csv_path = "/home/ws/src/hirise_data/rock_analysis_test.csv"             # The "Local" Map
    
    match_local_to_global(global_csv_path, local_csv_path)