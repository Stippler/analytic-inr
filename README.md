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
```

