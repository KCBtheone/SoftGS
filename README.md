# SoftGS

SoftGS edits Gaussian splats from ordered pairs of 3D control points. It fits bicubic charts to a source TSDF surface, solves for a smoothly blended displacement field with hard control constraints, and transports Gaussian centers, covariances and spherical harmonics. The source operator can be reused for new target locations of the same controls.

## Install

Use Python 3.9 or newer. Install the surface dependencies to build an operator. Applying an existing operator requires only the base dependencies.

```sh
python -m pip install ".[surface]"
```

## Build and edit

Supply a canonical 3DGS PLY, a triangle mesh extracted from a source TSDF, and two `(N, 3)` NumPy arrays of corresponding points in the model's world coordinate system. PLY files use log-scales, scalar-first quaternions and SH degrees 0–3. Source controls must constrain the affine modes in 3D.

```sh
softgs build --model source.ply --surface source_tsdf.ply \
    --source source_points.npy --output operator.npz

softgs edit --model source.ply --operator operator.npz \
    --target target_points.npy --output edited.ply
```

For source TSDF fusion from the PLY, install [PGSR](https://github.com/zju3dv/PGSR) with its CUDA rasterizer and make its repository importable in the same Python environment. Replace `--surface source_tsdf.ply` with `--cameras transforms_train.json`. The camera file uses Blender camera-to-world matrices and a horizontal field of view in radians. The first source image supplies the original resolution.

`softgs build --help` lists the view sampling, voxel resolution and chart parameters. The quadrature grid can be specified with `--grid-cells X Y Z`. For reproducing a run with an existing source integration grid, supply `--measure grid.npz` with `lower`, `width`, `spans` and `source_cell_measure_density`. The grid and its density are defined entirely in the canonical source frame. Existing source grids are needed to reproduce runs that used different grid settings from the CLI defaults.

Render the edited PLY with your 3DGS renderer at the desired cameras. For another edit, pass a new target array to the same operator and canonical PLY. The CLI reports infeasible controls and sampled fold or collapse checks. `--allow-unsafe-diagnostic` permits exporting a failed validity check for inspection where the covariance remains representable. It leaves the requested control displacement and field derivative intact.

## SoftBench

| Split | Source | Evaluation |
| --- | --- | --- |
| Single | Eight scenes adapted from NeuralEditor | One edit per scene, 90 target views each, full-image RGB metrics on white |
| Multi | Four objects with Blender lattice deformations | Fifteen edits, 75 target views each, target-foreground RGB metrics at native resolution |

Both splits use corresponding 3D control points at 16, 32, 64 and 128 points. Silhouette IoU thresholds predicted opacity and target alpha at 0.5. The paper averages views within each edit, then weights edits equally within each split.

The [SoftBench dataset repository](https://huggingface.co/datasets/KCB1/SoftBench) provides the prepared splits, including source checkpoints and integration grids, control pairs, cameras, target RGB and alpha, and the split-specific evaluation protocol.
