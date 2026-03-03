import csv
import numpy as np
import os
import sys

# Import your existing logic
# (Ensure your identifier.py has the fast hash builder we just wrote!)
from identifier import identify_geometric
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_match_result(global_pts, global_sizes, local_pts, local_sizes):
    """
    Plots the Global Map (Catalog) and Local Map (Observed), 
    scaling the shapes to represent actual physical rock footprints.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Draw Global Rocks (Solid gray ellipses)
    for i in range(len(global_pts)):
        x, y = global_pts[i]
        w, l = global_sizes[i]
        
        # Enforce a minimum visual footprint (e.g., 5cm) so tiny rocks don't vanish
        disp_w = max(w, 0.05)
        disp_l = max(l, 0.05)
        
        ellipse = Ellipse((x, y), width=disp_w, height=disp_l, 
                          color='lightgray', alpha=0.6, ec='gray')
        ax.add_patch(ellipse)

    # 2. Draw Local Rocks (Hollow orange dashed ellipses with an 'x' center)
    for i in range(len(local_pts)):
        x, y = local_pts[i]
        w, l = local_sizes[i]
        
        disp_w = max(w, 0.05)
        disp_l = max(l, 0.05)
        
        # Draw the outer boundary
        ellipse = Ellipse((x, y), width=disp_w, height=disp_l, 
                          color='none', ec='orange', lw=2, linestyle='--')
        ax.add_patch(ellipse)
        
        # Draw the exact centroid
        ax.scatter(x, y, c='orange', marker='x', s=30)

    # 3. Dummy plots for the legend
    ax.scatter([], [], c='lightgray', s=100, label='Global Map (Catalog)')
    ax.scatter([], [], facecolors='none', edgecolors='orange', s=100, linestyle='--', label='Local Rocks')

    # 4. Formatting
    ax.set_xlabel("Global X (meters)")
    ax.set_ylabel("Global Y (meters)")
    
    # CRITICAL: Ensures 1 meter on the X axis is exactly 1 meter on the Y axis
    ax.axis('equal') 
    
    # 5. Auto-scale the camera view to fit all points
    if len(global_pts) > 0 or len(local_pts) > 0:
        all_pts = np.vstack((global_pts, local_pts)) if len(local_pts) > 0 else global_pts
        min_x, max_x = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        min_y, max_y = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        
        # Add a 2-meter padding around the edges
        ax.set_xlim(min_x - 2.0, max_x + 2.0)
        ax.set_ylim(min_y - 2.0, max_y + 2.0)

    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def load_map_with_sizes(filepath):
    """
    Reads Map_X, Map_Y, Width_m, Length_m from ANY CSV that contains them.
    Ignores extra columns like Map_Z or Global_U.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None, None, None

    points = []
    sizes = []
    ids = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        # Clean up any accidental trailing spaces in the CSV headers
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        for row in reader:
            try:
                # These 4 columns exist in BOTH your global and local CSVs!
                x = float(row['Map_X'])
                y = float(row['Map_Y'])
                w = float(row['Width_m'])
                l = float(row['Length_m'])
                
                points.append([x, y])
                sizes.append([w, l]) 
                ids.append(row['ID'])
                
            except KeyError as e:
                # If a row is missing one of these exact columns, warn us
                print(f"Skipping row in {filepath}, missing column: {e}")
                continue
            except ValueError:
                # Skip rows with invalid/empty number formats
                continue

    return np.array(points, dtype=np.float32), np.array(sizes, dtype=np.float32), ids

def match_local_to_global(global_csv, local_csv):
    print(f"Loading Global Map: {global_csv}")
    global_pts, global_sizes, global_ids = load_map_with_sizes(global_csv)
    
    print(f"Loading Local Map:  {local_csv}")
    local_pts, local_sizes, local_ids = load_map_with_sizes(local_csv)

    if global_pts is None or local_pts is None: return
    
    global_pts[:, 1] = -global_pts[:, 1]
    
    # Pass the sizes into the updated plot function!
    # plot_match_result(global_pts, global_sizes, local_pts, local_sizes)

    match_percentage = 0.80  
    size_tolerance = 0.90    
    
    print(f"Points: Global={len(global_pts)}, Local={len(local_pts)}")
    print(f"Goal: Match at least {int(len(local_pts) * match_percentage)} rocks.")
    
    # --- FIX 1: Add observed_sizes to sim_input ---
    sim_input = {
        'observed_vectors': local_pts,
        'observed_sizes': local_sizes, 
        'n_false': int(len(local_pts) * (1.0 - match_percentage)) 
    }
    
    # --- FIX 2: Create a catalog dictionary with global_sizes ---
    catalog_dict = {
        'catalog_vectors': global_pts,
        'catalog_sizes': global_sizes   
    }

    result = identify_geometric(
        sim_input, 
        catalog_dict,             # <-- Pass the dictionary, not just the points
        eps=0.3,                  # Loosened slightly (30cm tolerance)
        binsize=0.01,             # Loosened from 0.001 (0.001 is too strict for real data!)
        ransac_iters=10000, 
        min_seed_inliers=5,       # 8 is very high; 5 (Triangle + 2 rocks) is a safe threshold
        early_exit_fraction=1.0,  # Force it to check everything to find the best match
        size_tolerance=size_tolerance # <-- Pass your size_tolerance variable!
    )
    best = result['best_solution']

    if best:
        # --- SIZE VERIFICATION STEP ---
        # The geometric matcher found a shape match. Now let's check if the sizes make sense.
        # This filters out "False Positives" where the shape is right but the rocks are wrong.
        
        matches = best['matches']
        
        success_ratio = len(matches) / len(local_pts)
        
        if success_ratio >= match_percentage:
            print("\n✅ ROBUST MATCH CONFIRMED!")
            print(f"Geometric Inliers: {len(matches)}")
            print(f"Size-Verified Inliers: {len(matches)} ({success_ratio*100:.1f}%)")
            
            # Recalculate pose might be needed if we dropped too many outliers, 
            # but usually the original R/t is fine if the geometric consensus was strong.
            t = best['t']
            R = best['R']
            theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            print(f"\n--- ESTIMATED POSE ---")
            print(f"Rotation:    {theta_deg:.2f} degrees")
            print(f"Translation: X={t[0]:.2f}, Y={t[1]:.2f}")
            
            print("\n--- CONFIRMED MATCHES ---")
            for (c, o, r) in matches:
                 print(f"  Local Rock {local_ids[o]} -> Global Rock {global_ids[c]}")
                 
            s = best['s']  # Get the scale factor
            
            # Apply the Similarity Transform to the Local Points: X_new = s * (R @ X) + t
            # We transpose local_pts to correctly multiply with the 2x2 Rotation matrix
            transformed_local_pts = (s * np.dot(R, local_pts.T)).T + t
            
            # Scale the local rock dimensions so they visually match the global map
            transformed_local_sizes = local_sizes * s
            
            print("\nDrawing the aligned map match...")
            plot_match_result(global_pts, global_sizes, transformed_local_pts, transformed_local_sizes)
                 
        else:
            print(f"\n⚠️ Geometric match found, but failed size verification.")
            print(f"Valid Size Matches: {len(matches)}/{len(local_pts)} ({success_ratio*100:.1f}%)")
            print("Action: Try increasing 'size_tolerance' or checking your segmentation scale.")
            t = best['t']
            R = best['R']
            theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            s = best['s']  # Get the scale factor

            transformed_local_pts = (s * np.dot(R, local_pts.T)).T + t
            transformed_local_sizes = local_sizes * s
            
            print("\nDrawing the aligned map match...")
            plot_match_result(global_pts, global_sizes, transformed_local_pts, transformed_local_sizes)

    else:
        print("\n❌ NO MATCH FOUND.")
        print("Tips:")
        print("1. Increase 'eps' (tolerance).")
        print("2. Increase 'match_percentage' (maybe you only see 50% of the rocks?).")


if __name__ == "__main__":
    # Replace with your actual file paths
    global_csv_path = "/home/ws/src/hirise_data/above_rock_analysis.csv" # The "Above" Map
    local_csv_path = "/home/ws/src/hirise_data/rock_analysis_test4.csv"             # The "Local" Map
    
    match_local_to_global(global_csv_path, local_csv_path)