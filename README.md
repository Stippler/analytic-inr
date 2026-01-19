# Installation

For typical configuration:
```bash
conda create -n spline python=3.11 -y
conda install pytorch3d -c pytorch3d
pip install click matplotlib trimesh rtree
```

For RTX 5090 support:
```bash
conda create -n spline python=3.11 -y
sudo apt update
sudo apt install -y build-essential
sudo apt install -y nvidia-cuda-toolkit
```

Then cd to some directory where you can have a repo that you leave there.

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . --no-build-isolation
pip install click matplotlib trimesh rtree
pip install trame trame-vuetify trame-vtk
```

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

<details>
<summary>Launch Configs</summary>

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Demo",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/demo.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Preprocess: Armadillo",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/preprocess.py",
            "args": [
                "--mesh", "Stanford_armadillo"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Train: Armadillo",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train_sdf.py",
            "args": [
                "--task", "from_sdf",
                "--arch", "relu_mlp_d4_w128",
                "--mesh", "Stanford_armadillo",
                "--n_iters", "1000"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Eval: Armadillo",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/evaluate.py",
            "args": [
                "--task", "from_sdf",
                "--arch", "relu_mlp_d4_w256",
                "--mesh", "Stanford_armadillo",
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Eval: All",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/evaluate.py",
            "args": [
                "--task", "from_sdf",
                // "--arch", "relu_mlp_d4_w256",
                // "--mesh", "Stanford_armadillo",
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Metrics: All",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/accumulate_metrics.py",
            "args": [
                "--task", "from_sdf",
                // "--arch", "relu_mlp_d4_w256",
                // "--mesh", "Stanford_armadillo",
            ],
            "console": "integratedTerminal"
        }
    ]
}
```
</details>

<details>
<summary>How to eval our models</summary>

In your `.pt`-file, you also need the model specification, which you can get by 
`ReluMLP::config()`. It is now stored in the state dict.

```json
{
    "name": "Eval: Ours",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/evaluate.py",
    "args": [
        "--sdf_path", "outputs/Stanford_armadillo/20260119_110333/sdf_epoch_0000.pt",
        "--mesh", "Stanford_armadillo",
    ],
    "console": "integratedTerminal"
},
```
</details>