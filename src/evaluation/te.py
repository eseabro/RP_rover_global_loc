import sys
sys.path.insert(0, '/home/ws/evaluation')
from eval_stereo import read_bag, lidar_depth_for_rock

frames = read_bag('/home/ws/rock_eval')
f = frames[0]
print(f"lidar_pts shape: {f.lidar_pts.shape}")
print(f"n_rocks: {len(f.rock_obs)}")

for i, rock in enumerate(f.rock_obs[:3]):
    c = rock['centroid']
    print(f"\nrock[{i}] centroid type={type(c)} shape={c.shape} values={c}")
    stereo_depth = float(c[2])
    print(f"  stereo_depth={stereo_depth:.3f} valid={0.1 < stereo_depth < 5.0}")
    result = lidar_depth_for_rock(f.lidar_pts, c, search_radius_m=0.8)
    print(f"  lidar_depth={result}")
    
    # Manual check
    cx_lid = -c[2]; cy_lid = -c[0]
    import numpy as np
    dxy = np.sqrt((f.lidar_pts[:,0]-cx_lid)**2 + (f.lidar_pts[:,1]-cy_lid)**2)
    nearby = f.lidar_pts[dxy < 0.8]
    print(f"  manual nearby={len(nearby)} min_dxy={dxy.min():.4f}")