"""EgoVerse Trajectory Diversity & Curriculum Engine (Track 2).

Pipeline stages:
    loader           -> read end-effector trajectories out of .zarr episode stores
    diversity_engine -> pairwise Dynamic Time Warping distance matrix
    cluster_mapper   -> UMAP projection, precomputed-distance clustering, metrics
    curriculum       -> kinematic difficulty scoring and training-order synthesis
"""

__version__ = "1.0.0"

__all__ = ["loader", "diversity_engine", "cluster_mapper", "curriculum"]
