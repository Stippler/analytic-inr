# Spline-based Construction of INRs

### Setup

First, convert the mesh (`*.ply`) into sdf point clouds
```bash
python scripts/preprocess.py --mesh <MESH_NAME>
```
e.g. `python scripts/preprocess.py --mesh Stanford_armadillo` yields `data/sdf_point_clouds/Stanford_armadillo.ply`.

#### Training
```bash
python scripts/train_sdf.py --task from_sdf --arch relu_mlp_d4_w128 --mesh <MESH_NAME> --n_iters 1000
```

#### Evaluation
```bash
python scripts/evaluate.py --task from_sdf --arch relu_mlp_d4_w128 --mesh <MESH_NAME>
```
which yields the following results:
```json
{
    "chamfer_l1": 0.008942771703004837,
    "hausdorff": 0.14315305650234222,
    "dist_gt_to_neural_mean": 0.0034248116426169872,
    "dist_neural_to_gt_mean": 0.014460732229053974,
    "emd_loss": 0.013952197507023811,
    "details": {
        "arch": "relu_mlp_d4_w128",
        "mesh": "Stanford_armadillo",
    }
}
```
as well as a mesh in `nets/from_sdf/relu_mlp_d4_w128/Stanford_armadillo/mesh.ply`.