"""Read and write standard 3D Gaussian Splatting PLY files."""

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def sh_degree(vertices):
    names = set(vertices.dtype.names)
    rest = {name for name in names if name.startswith("f_rest_")}
    for degree in range(4):
        expected = {f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))}
        if rest == expected:
            return degree
    raise ValueError(
        "The model must contain a complete SH basis of degree 0 through 3."
    )


def read_model(path):
    ply = PlyData.read(str(path), mmap=False)
    vertex = ply["vertex"].data
    scale_names = [f"scale_{i}" for i in range(3)]
    rotation_names = [f"rot_{i}" for i in range(4)]
    required = [
        *list("xyz"),
        "opacity",
        *scale_names,
        *rotation_names,
        *(f"f_dc_{i}" for i in range(3)),
    ]
    missing = set(required) - set(vertex.dtype.names)
    if missing or not len(vertex):
        raise ValueError(
            f"Invalid Gaussian PLY. Missing attributes: {sorted(missing)}."
        )
    sh_degree(vertex)
    required.extend(name for name in vertex.dtype.names if name.startswith("f_rest_"))
    if not all(np.isfinite(vertex[name]).all() for name in required):
        raise ValueError("The Gaussian PLY contains nonfinite attributes.")
    positions = np.column_stack([vertex[axis] for axis in "xyz"]).astype(float)
    scales = np.column_stack([vertex[name] for name in scale_names]).astype(float)
    quaternions = np.column_stack([vertex[name] for name in rotation_names]).astype(
        float
    )
    rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    return ply, positions, scales, rotation


def write_model(stream, source, vertices):
    original = source["vertex"]
    replacement = PlyElement.describe(
        vertices,
        "vertex",
        comments=original.comments,
        len_types={
            p.name: p.len_dtype for p in original.properties if hasattr(p, "len_dtype")
        },
    )
    elements = [
        replacement if element.name == "vertex" else element
        for element in source.elements
    ]
    PlyData(
        elements,
        text=source.text,
        byte_order=source.byte_order,
        comments=source.comments,
        obj_info=source.obj_info,
    ).write(stream)


def sample_support(positions, scales, rotation, samples=512):
    indices = np.linspace(
        0, len(positions) - 1, min(len(positions), samples), dtype=int
    )
    axes = (rotation[indices] * np.exp(scales[indices, None, :])).transpose(0, 2, 1)
    offsets = 3 * np.concatenate((axes, -axes), axis=1)
    return (positions[indices, None] + offsets).reshape(-1, 3)
