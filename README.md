# Spline-based Construction of INRs

### Setup

First, convert the mesh (`*.ply`) into sdf point clouds
```bash
python scripts/preprocess.py --mesh <MESH_NAME>
```
e.g. `python scripts/preprocess.py --mesh Armadillo` yields `data/sdf_point_clouds/Armadillo.ply`.
