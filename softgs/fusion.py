"""Fuse standard 3DGS depth maps from source cameras into a TSDF mesh."""

import json
import math
from pathlib import Path

import numpy as np


def fuse_source(
    positions,
    log_scales,
    vertices,
    transforms_path,
    out,
    diagonal,
    *,
    view_count=64,
    image_limit=512,
    voxels=256,
):
    import open3d as o3d
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise ValueError(
            "Source TSDF fusion requires CUDA. Supply --surface to use a fused mesh."
        )
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:
        raise ValueError(
            "Source TSDF fusion requires Graphdeco's 3DGS CUDA rasterizer "
            "(diff-gaussian-rasterization)."
        ) from exc
    with transforms_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    frames = metadata["frames"]
    angle = float(metadata["camera_angle_x"])
    if not frames or not 0 < angle < math.pi:
        raise ValueError(
            "The source camera file has no views or an invalid horizontal field of view."
        )
    if "w" in metadata and "h" in metadata:
        original_width, original_height = int(metadata["w"]), int(metadata["h"])
    else:
        image_path = transforms_path.parent / Path(frames[0]["file_path"])
        if not image_path.suffix:
            image_path = image_path.with_suffix(".png")
        with Image.open(image_path) as image:
            original_width, original_height = image.size
    if min(original_width, original_height) <= 0:
        raise ValueError("Source image dimensions must be positive.")
    ratio = min(1.0, image_limit / max(original_width, original_height))
    width = max(1, round(ratio * original_width))
    height = max(1, round(ratio * original_height))
    focal = 0.5 * original_width / math.tan(angle / 2)
    fx = focal * width / original_width
    fy = focal * height / original_height
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, width / 2, height / 2
    )
    tanfovx, tanfovy = width / (2 * fx), height / (2 * fy)
    near, far = 0.01, 1000.0
    device = torch.device("cuda")
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    # Graphdeco's rasterizer reads the transposed view and projection matrices.
    projection[0, 0] = 1 / tanfovx
    projection[1, 1] = 1 / tanfovy
    projection[2, 2] = far / (far - near)
    projection[2, 3] = 1
    projection[3, 2] = -far * near / (far - near)
    voxel_size = diagonal / voxels
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    means3d = torch.as_tensor(positions, dtype=torch.float32, device=device)
    means2d = torch.zeros_like(means3d)
    scales = torch.as_tensor(log_scales, dtype=torch.float32, device=device).exp()
    quaternions = np.column_stack([vertices[f"rot_{i}"] for i in range(4)])
    rotations = torch.nn.functional.normalize(
        torch.as_tensor(quaternions, dtype=torch.float32, device=device), dim=1
    )
    opacities = torch.as_tensor(
        vertices["opacity"].copy(), dtype=torch.float32, device=device
    ).sigmoid()[:, None]
    colors = torch.zeros_like(means3d)
    colors[:, 1] = 1
    raster_options = dict(
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        sh_degree=0,
        prefiltered=False,
        debug=False,
    )
    # The 3DGS antialiasing build exposes the option; the original build does not.
    if "antialiasing" in GaussianRasterizationSettings._fields:
        raster_options["antialiasing"] = False
    dummy_rgb = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
    chosen = np.linspace(0, len(frames) - 1, min(view_count, len(frames)), dtype=int)
    with torch.inference_mode():
        for frame_index in chosen:
            c2w = np.array(frames[frame_index]["transform_matrix"], dtype=float)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                raise ValueError(f"Invalid transform in source frame {frame_index}.")
            c2w[:3, 1:3] *= -1  # Blender camera axes to positive-z camera axes.
            w2c = np.linalg.inv(c2w)
            view = torch.as_tensor(w2c.T.copy(), dtype=torch.float32, device=device)
            settings = GaussianRasterizationSettings(
                viewmatrix=view,
                projmatrix=view @ projection,
                campos=torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=device),
                **raster_options,
            )
            # Linear color channels accumulate weighted view-space z and coverage.
            colors[:, 0] = means3d @ view[:3, 2] + view[3, 2]
            render, *_ = GaussianRasterizer(raster_settings=settings)(
                means3D=means3d,
                means2D=means2d,
                opacities=opacities,
                colors_precomp=colors,
                scales=scales,
                rotations=rotations,
            )
            weighted_depth, alpha = render[:2].cpu().numpy()
            depth = np.zeros_like(weighted_depth)
            np.divide(weighted_depth, alpha, out=depth, where=alpha > 0.5)
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                continue
            depth[~valid] = 0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                dummy_rgb,
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,
                depth_trunc=float(depth.max() + diagonal),
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsics, w2c)
    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) < 16 or not len(mesh.triangles):
        raise ValueError("The fused TSDF has insufficient surface geometry.")
    if not o3d.io.write_triangle_mesh(str(out), mesh):
        raise OSError(f"Could not write {out}.")
