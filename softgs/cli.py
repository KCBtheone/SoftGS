"""Build a source response and apply sparse point edits."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np

from .atlas import build_atlas, check_cover
from .basis import evaluate
from .fusion import fuse_source
from .model import read_model, sample_support, write_model
from .solve import gaussian_measure, integrate, prepare_response, quadrature
from .transport import transport


def point_pairs(path):
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{path} must be a .npy file.")
    points = np.load(path, allow_pickle=False)
    if points.dtype.kind not in "fiu":
        raise ValueError(f"{path} must contain real coordinates.")
    points = points.astype(float, copy=False)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError(f"{path} must be a nonempty (N, 3) NumPy array.")
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains nonfinite points.")
    return points


def source_measure(args, positions, scales, opacity, source):
    if args.measure is None:
        return gaussian_measure(
            positions, scales, opacity, source, np.array(args.grid_cells)
        )
    with np.load(args.measure, allow_pickle=False) as measure:
        lower, width, spans, density = (
            measure[name]
            for name in ("lower", "width", "spans", "source_cell_measure_density")
        )
    if spans.ndim == 0:
        spans = np.repeat(spans, 3)
    if (
        lower.shape != (3,)
        or width.shape != (3,)
        or spans.shape != (3,)
        or not np.issubdtype(spans.dtype, np.integer)
        or np.any(spans <= 0)
        or np.any(width <= 0)
        or density.size != np.prod(spans)
        or np.any(density <= 0)
        or not np.isfinite(np.r_[lower, width, density.ravel()]).all()
    ):
        raise ValueError("Invalid source integration grid or density.")
    return lower, width, spans, density.ravel()


def write_output(path, writer):
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.close()
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def build(args):
    model, positions, scales, rotation = read_model(args.model)
    vertices = model["vertex"].data
    source = point_pairs(args.source)
    if len(source) < 4:
        raise ValueError(
            "At least four source controls are required to constrain the affine modes."
        )
    diagonal = np.linalg.norm(np.ptp(positions, axis=0))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("The source Gaussian centers must have finite extent.")
    with TemporaryDirectory(dir=args.output.parent, prefix=".softgs-") as scratch:
        surface = args.surface
        if surface is None:
            surface = Path(scratch) / "surface.ply"
            fuse_source(
                positions,
                scales,
                vertices,
                args.cameras,
                surface,
                diagonal,
                view_count=args.source_views,
                image_limit=args.image_limit,
                voxels=args.voxels,
            )
        atlas = build_atlas(
            surface, args.charts_per_diagonal, args.support_spacings, args.fit_spacings
        )
    measure = source_measure(args, positions, scales, vertices["opacity"], source)
    sites, volume, density = quadrature(*measure)
    check_cover(atlas, positions, "Gaussian center")
    check_cover(atlas, source, "source control")
    check_cover(atlas, sites, "quadrature")
    check_cover(atlas, sample_support(positions, scales, rotation), "Gaussian support")
    bending, mass = integrate(
        atlas, sites, volume, density, gram_device=args.gram_device
    )
    response = prepare_response(atlas, source, mass, bending)
    write_output(
        args.output,
        lambda stream: np.savez(
            stream,
            **atlas,
            source=source,
            response=response,
            canonical_centers=positions,
        ),
    )
    print(
        f"Wrote {len(atlas['centers'])} charts and {len(source)} controls to {args.output}"
    )


def check_edit(jacobians, support_jacobians, rotation, scales):
    stretches = np.linalg.svd(jacobians, compute_uv=False)
    support_stretches = np.linalg.svd(support_jacobians, compute_uv=False)
    center_bad = (np.linalg.det(jacobians) <= 0) | (stretches[:, -1] <= 1e-4)
    support_bad = (np.linalg.det(support_jacobians) <= 0) | (
        support_stretches[:, -1] <= 1e-4
    )
    order = np.argsort(scales, axis=1)[:, ::-1]
    ordered = np.take_along_axis(scales, order, axis=1)
    thin = ordered[:, 2] - ordered[:, 1] <= np.log(1e-3)
    tangent = np.take_along_axis(rotation[thin], order[thin, None, :2], axis=2)
    tangent_bad = (
        np.linalg.svd(jacobians[thin] @ tangent, compute_uv=False)[:, -1] <= 1e-4
    )
    if not np.any(center_bad) and not np.any(support_bad) and not np.any(tangent_bad):
        return None
    return (
        f"{center_bad.sum()} folded or collapsed centers, {support_bad.sum()} failed "
        f"support probes, {tangent_bad.sum()} collapsed thin Gaussian supports"
    )


def edit(args):
    model, canonical, scales, rotation = read_model(args.model)
    with np.load(args.operator, allow_pickle=False) as saved:
        atlas = {key: saved[key] for key in ("centers", "radii", "axes", "controls")}
        source = saved["source"]
        response = saved["response"]
        expected = saved["canonical_centers"]
    if canonical.shape != expected.shape or not np.allclose(
        canonical, expected, rtol=0, atol=1e-7
    ):
        raise ValueError("The source response belongs to a different canonical model.")
    target = point_pairs(args.target)
    if source.shape != target.shape:
        raise ValueError(
            f"The operator expects {len(source)} target controls in source order."
        )
    coefficient = response @ (target - source)
    moved, jacobians = evaluate(atlas, coefficient, canonical)
    controls, _ = evaluate(atlas, coefficient, source)
    error = np.linalg.norm(controls - target, axis=1).max()
    if not np.isfinite(error) or error >= 1e-5:
        raise ValueError(f"The edit violates hard controls (error {error:.3g}).")
    _, support = evaluate(
        atlas, coefficient, sample_support(canonical, scales, rotation)
    )
    if not all(np.isfinite(values).all() for values in (moved, jacobians, support)):
        raise ValueError("The deformation field contains nonfinite values.")
    failure = check_edit(jacobians, support, rotation, scales)
    if failure and not args.allow_unsafe_diagnostic:
        raise ValueError(f"The deformation did not pass validity checks: {failure}.")
    if failure:
        print(f"Diagnostic edit with invalid deformation: {failure}.", file=sys.stderr)
    result = transport(model["vertex"].data, moved, jacobians, rotation, scales)
    write_output(args.output, lambda stream: write_model(stream, model, result))
    print(
        f"Wrote {len(moved)} edited Gaussians to {args.output} (control error {error:.3g})"
    )


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite value.")
    return number


def count(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("build", help="Build a reusable source operator")
    apply = commands.add_parser(
        "edit", help="Apply an edit to the canonical Gaussian model"
    )
    for command in (prepare, apply):
        command.add_argument(
            "--model", type=Path, required=True, help="Canonical Gaussian PLY"
        )
        command.add_argument("--output", type=Path, required=True)
    geometry = prepare.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--surface", type=Path, help="Source TSDF mesh")
    geometry.add_argument(
        "--cameras", type=Path, help="Source Blender camera JSON for 3DGS depth fusion"
    )
    prepare.add_argument(
        "--source", type=Path, required=True, help="Ordered source controls (.npy)"
    )
    measure = prepare.add_mutually_exclusive_group()
    measure.add_argument(
        "--measure", type=Path, help="Precomputed source integration grid (.npz)"
    )
    measure.add_argument("--grid-cells", type=count, nargs=3, default=(3, 3, 3))
    prepare.add_argument("--source-views", type=count, default=64)
    prepare.add_argument("--image-limit", type=count, default=512)
    prepare.add_argument(
        "--voxels", type=count, default=256, help="Voxels per source diagonal"
    )
    prepare.add_argument("--charts-per-diagonal", type=positive, default=6)
    prepare.add_argument("--support-spacings", type=positive, default=4)
    prepare.add_argument("--fit-spacings", type=positive, default=1.5)
    prepare.add_argument(
        "--gram-device", default="cpu", help="PyTorch device for large Gram products"
    )
    apply.add_argument(
        "--operator", type=Path, required=True, help="Source operator (.npz)"
    )
    apply.add_argument(
        "--target", type=Path, required=True, help="Ordered target controls (.npy)"
    )
    apply.add_argument(
        "--allow-unsafe-diagnostic",
        action="store_true",
        help="Export a diagnostic edit when support checks fail",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.model.resolve():
        parser.error("The output must differ from the canonical model path.")
    input_files = (
        (args.source, args.surface, args.cameras, args.measure)
        if args.command == "build"
        else (args.operator, args.target)
    )
    if any(path and path.resolve() == args.output.resolve() for path in input_files):
        parser.error("The output must differ from every input path.")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (build if args.command == "build" else edit)(args)
    except (ImportError, KeyError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"softgs: {exc}\n")


if __name__ == "__main__":
    main()
