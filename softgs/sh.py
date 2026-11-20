"""Real spherical harmonics in the 3D Gaussian Splatting PLY convention."""

import numpy as np


def eval_sh(degree, coefficients, directions):
    x, y, z = np.moveaxis(directions, -1, 0)
    values = [np.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        c = 0.4886025119029199
        values.extend((-c * y, c * z, -c * x))
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        values.extend(
            (
                1.0925484305920792 * x * y,
                -1.0925484305920792 * y * z,
                0.31539156525252005 * (2 * zz - xx - yy),
                -1.0925484305920792 * x * z,
                0.5462742152960396 * (xx - yy),
            )
        )
    if degree >= 3:
        values.extend(
            (
                -0.5900435899266435 * y * (3 * xx - yy),
                2.890611442640554 * x * y * z,
                -0.4570457994644658 * y * (4 * zz - xx - yy),
                0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                -0.4570457994644658 * x * (4 * zz - xx - yy),
                1.445305721320277 * z * (xx - yy),
                -0.5900435899266435 * x * (xx - 3 * yy),
            )
        )
    return np.stack(values, axis=-1) @ coefficients.T
