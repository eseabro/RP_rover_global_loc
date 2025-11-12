"""pyramid_star_id: Pyramid-based star identification package

Exposes high-level functions:
  - build_catalog(...)
  - simulate_observations(...)
  - precompute_catalog_pyramids(...)
  - identify(...)
"""
from .catalog import build_catalog, build_mars_catalog, load_catalog, save_catalog_csv
from .simulator import simulate_observations, simulate_identity_observations, simulate_observations_with_pose
from .pyramids import precompute_catalog_pyramids, pyramid_signature_from_vectors, build_kvector, query_kvector
from .identifier import identify
__all__ = ["build_catalog", "load_catalog", "save_catalog_csv", "simulate_observations",
           "precompute_catalog_pyramids", "pyramid_signature_from_vectors",
           "identify", "build_kvector", "query_kvector", "simulate_identity_observations", "simulate_observations_with_pose",
           "build_mars_catalog"]
