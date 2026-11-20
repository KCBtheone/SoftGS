"""Construct a bicubic surface atlas from source TSDF geometry."""

import numpy as np
from scipy.spatial import cKDTree

from .basis import bernstein


def build_atlas(mesh_path, charts_per_diagonal=6, support_spacings=4, fit_spacings=1.5):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) < 16 or not len(mesh.triangles):
        raise ValueError(
            "The source TSDF mesh must contain triangles and at least 16 vertices."
        )
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("The source TSDF mesh must have finite extent.")
    spacing = diagonal / charts_per_diagonal
    radius = support_spacings * spacing
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    sampled = np.asarray(cloud.voxel_down_sample(spacing).points)
    tree = cKDTree(points)
    selected = np.unique(tree.query(sampled)[1])
    centers = points[selected]
    grid = np.indices((4, 4)).reshape(2, -1).T / 3
    frames = []
    controls = []
    for index, center in zip(selected, centers):
        normal = normals[index]
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("A source chart center has no usable surface normal.")
        normal = normal / length
        tangent = np.cross(normal, np.eye(3)[np.abs(normal).argmin()])
        tangent /= np.linalg.norm(tangent)
        frame = np.stack((tangent, np.cross(normal, tangent), normal))
        neighbors = tree.query_ball_point(center, fit_spacings * spacing)
        if len(neighbors) < 16:
            neighbors = tree.query(center, k=min(64, len(points)))[1]
        relative = (points[neighbors] - center) @ frame.T
        u, v = ((relative[:, :2] / radius + 1) / 2).T
        basis = np.einsum("ni,nj->nij", bernstein(u), bernstein(v)).reshape(-1, 16)
        height = np.linalg.lstsq(basis, relative[:, 2], rcond=None)[0]
        control = (
            center + ((2 * grid - 1) * radius) @ frame[:2] + height[:, None] * normal
        )
        frames.append(frame)
        controls.append(control.reshape(4, 4, 3))
    return dict(
        centers=centers,
        radii=np.full(len(centers), radius),
        axes=np.stack(frames),
        controls=np.stack(controls),
    )


def check_cover(atlas, points, label):
    # Atlas construction uses one shared radius.
    distances = cKDTree(atlas["centers"]).query(points)[0]
    missing = np.count_nonzero(distances >= atlas["radii"][0])
    if missing:
        raise ValueError(f"The source atlas does not cover {missing} {label} points.")
