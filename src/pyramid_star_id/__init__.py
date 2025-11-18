"""pyramid_star_id: Pyramid-based star identification package

Exposes high-level functions:
  - build_catalog(...)
  - simulate_observations(...)
  - precompute_catalog_pyramids(...)
  - identify(...)
"""
from .catalog import build_catalog, load_catalog, save_catalog_csv
from .simulator import simulate_observations, simulate_identity_observations, simulate_observations_with_pose, sample_observed
from .pyramids import precompute_catalog_pyramids, pyramid_signature_from_vectors, build_kvector, query_kvector
from .identifier import identify_geometric, catalog_to_pts2d, build_geometric_hash_from_pts
from .geometry import kabsch_rotation, apply_rotation
from .hash import build_geometric_hash, query_geometric_hash
__all__ = ["build_catalog", "load_catalog", "save_catalog_csv", "simulate_observations",
           "precompute_catalog_pyramids", "pyramid_signature_from_vectors",
           "build_kvector", "query_kvector", "simulate_identity_observations", "simulate_observations_with_pose",
           "sample_observed", "kabsch_rotation", "apply_rotation", "build_geometric_hash_from_pts",
           "build_geometric_hash", "query_geometric_hash", "identify_geometric", "catalog_to_pts2d"]
