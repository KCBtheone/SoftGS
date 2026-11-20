"""Fuse expected depths rendered from source cameras into a TSDF mesh."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def fuse_source(
    model_path,
    transforms_path,
    out,
    diagonal,
    degree,
    *,
    view_count=64,
    image_limit=512,
    voxels=256,
):
    import open3d as o3d
    import torch
    from PIL import Image

    try:
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import getProjectionMatrix
    except ImportError as exc:
        raise ValueError(
            "Source TSDF fusion requires PGSR and its built CUDA rasterizer."
        ) from exc
    if not torch.cuda.is_available():
        raise ValueError(
            "Source TSDF fusion requires CUDA. Supply --surface to use a fused mesh."
        )
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
    fovx, fovy = 2 * math.atan(width / (2 * fx)), 2 * math.atan(height / (2 * fy))
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, width / 2, height / 2
    )
    projection = getProjectionMatrix(
        znear=0.01, zfar=1000.0, fovX=fovx, fovY=fovy
    ).T.cuda()
    voxel_size = diagonal / voxels
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    model = GaussianModel(sh_degree=degree)
    model.load_ply(str(model_path))
    pipeline = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    background = torch.zeros(3, device="cuda")
    colors = torch.zeros((len(model.get_xyz), 3), device="cuda")
    colors[:, 1] = 1
    dummy_rgb = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
    chosen = np.linspace(0, len(frames) - 1, min(view_count, len(frames)), dtype=int)
    with torch.inference_mode():
        for frame_index in chosen:
            c2w = np.array(frames[frame_index]["transform_matrix"], dtype=float)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                raise ValueError(f"Invalid transform in source frame {frame_index}.")
            c2w[:3, 1:3] *= -1  # Blender camera axes to positive-z camera axes.
            w2c = np.linalg.inv(c2w)
            view = torch.as_tensor(w2c.T, dtype=torch.float32, device="cuda")
            camera = SimpleNamespace(
                FoVx=fovx,
                FoVy=fovy,
                image_width=width,
                image_height=height,
                world_view_transform=view,
                full_proj_transform=view @ projection,
                camera_center=torch.as_tensor(
                    c2w[:3, 3], dtype=torch.float32, device="cuda"
                ),
            )
            # Render the weighted depth and the coverage with the same splats.
            colors[:, 0] = model.get_xyz @ view[:3, 2] + view[3, 2]
            render_result = render(
                camera,
                model,
                pipeline,
                background,
                override_color=colors,
                return_plane=False,
                return_depth_normal=False,
            )["render"]
            alpha = render_result[1]
            depth = (render_result[0] / alpha.clamp_min(1e-8)).cpu().numpy()
            valid = (alpha.cpu().numpy() > 0.5) & np.isfinite(depth) & (depth > 0)
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
