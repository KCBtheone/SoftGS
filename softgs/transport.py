"""Transport Gaussian covariances and view-dependent color through an edit."""

import numpy as np
from scipy.spatial.transform import Rotation

from .sh import eval_sh


class SHTransport:
    """Rotate each SH band in the renderer's own real basis."""

    def __init__(self, degree):
        self.degree = degree
        count = (degree + 1) ** 2
        i = np.arange(count)
        z = 1 - 2 * (i + 0.5) / len(i)
        radius = np.sqrt(1 - z * z)
        azimuth = np.pi * (3 - np.sqrt(5)) * i
        self.directions = np.column_stack(
            [radius * np.cos(azimuth), radius * np.sin(azimuth), z]
        )
        self.bands = tuple(
            slice(band * band, (band + 1) ** 2) for band in range(1, degree + 1)
        )
        self.basis_coefficients = np.eye(count)
        basis = eval_sh(degree, self.basis_coefficients, self.directions)
        self.band_inverse = [np.linalg.pinv(basis[:, band]) for band in self.bands]

    def rotate(self, sh, rotation):
        # In column convention the old viewing direction is Q^T d.
        basis = eval_sh(
            self.degree, self.basis_coefficients, self.directions @ rotation
        )
        result = sh.copy()
        for band, inverse in zip(self.bands, self.band_inverse):
            result[:, band] = inverse @ (basis[:, :, band] @ sh[:, band])
        return result


def covariance_factor(jacobian, rotation, log_scale):
    """Recover J Sigma J^T for thin and full-rank source Gaussians."""
    linear = jacobian @ rotation
    shift = log_scale.max(axis=1)
    factor = linear * np.exp(log_scale - shift[:, None])[:, None, :]
    axes, singular, _ = np.linalg.svd(factor)
    first = shift + np.log(singular[:, 0])
    second = np.empty_like(first)
    thin_plane = singular[:, 1] > 1e-10 * singular[:, 0]
    second[thin_plane] = shift[thin_plane] + np.log(singular[thin_plane, 1])

    # In a near-rank-one factor, obtain s1*s2 and the minor axis from the cofactor.
    line = ~thin_plane
    if line.any():
        h = linear[line]
        cofactor = np.stack(
            [
                np.cross(h[:, :, 1], h[:, :, 2]),
                np.cross(h[:, :, 2], h[:, :, 0]),
                np.cross(h[:, :, 0], h[:, :, 1]),
            ],
            axis=-1,
        )
        pair_log = log_scale[line].sum(axis=1)[:, None] - log_scale[line]
        area_shift = pair_log.max(axis=1)
        area = cofactor * np.exp(pair_log - area_shift[:, None])[:, None, :]
        area_axes, area_singular, _ = np.linalg.svd(area)
        second[line] = area_shift + np.log(area_singular[:, 0]) - first[line]
        e1 = axes[line, :, 0]
        e3 = area_axes[:, :, 0] - (area_axes[:, :, 0] * e1).sum(axis=1)[:, None] * e1
        e3 /= np.linalg.norm(e3, axis=1)[:, None]
        axes[line] = np.stack([e1, np.cross(e3, e1), e3], axis=-1)

    # det(JR diag(s)) fixes the minor scale without taking log of a rounded zero.
    sign, log_det = np.linalg.slogdet(linear)
    if np.any(sign <= 0):
        raise ValueError("Covariance transport requires a positive orientation")
    third = log_det + log_scale.sum(axis=1) - first - second
    axes[:, :, -1] *= np.linalg.det(axes)[:, None]
    return (
        axes,
        np.column_stack([first, second, third]),
        int(np.count_nonzero(singular[:, 2] == 0)),
    )


def covariance_polar(jacobian, source_rotation, source_log_scale):
    """Minimize E[||J delta-Q delta||^2], delta~N(0, canonical Sigma)."""
    relative_variance = np.exp(
        2 * (source_log_scale - source_log_scale.max(axis=1)[:, None])
    )
    covariance = (
        source_rotation * relative_variance[:, None, :]
    ) @ source_rotation.transpose(0, 2, 1)
    left, _, right = np.linalg.svd(jacobian @ covariance)
    left[np.linalg.det(left @ right) < 0, :, -1] *= -1
    return left @ right


def transport(vertices, positions, jacobians, rotation, scales, batch_size=8192):
    """Update centers, covariance factors and SH while preserving other attributes."""
    from .model import sh_degree

    degree = sh_degree(vertices)
    sh = SHTransport(degree)
    sh_names = [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    result = vertices.copy()
    for start in range(0, len(vertices), batch_size):
        select = slice(start, start + batch_size)
        source_rotation = rotation[select].copy()
        source_scale = scales[select]
        jacobian = jacobians[select]
        # Flipping one axis of the factor leaves its covariance unchanged.
        source_rotation[np.linalg.det(jacobian) < 0, :, -1] *= -1
        axes, new_scale, _ = covariance_factor(jacobian, source_rotation, source_scale)
        quat = Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]]
        attributes = [
            (tuple("xyz"), positions[select]),
            ([f"rot_{i}" for i in range(4)], quat),
            ([f"scale_{i}" for i in range(3)], new_scale),
        ]
        if degree:
            count = (degree + 1) ** 2
            coefficients = np.zeros((len(quat), count, 3))
            coefficients[:, 1:] = (
                np.column_stack([vertices[name][select] for name in sh_names])
                .reshape(-1, 3, count - 1)
                .transpose(0, 2, 1)
            )
            polar = covariance_polar(jacobian, source_rotation, source_scale)
            rotated = sh.rotate(coefficients, polar)
            attributes.append(
                (sh_names, rotated[:, 1:].transpose(0, 2, 1).reshape(len(quat), -1))
            )
        for names, values in attributes:
            if not np.isfinite(values).all():
                raise ValueError(
                    "The edited Gaussian attributes contain nonfinite values."
                )
            for i, name in enumerate(names):
                result[name][select] = values[:, i]
                if not np.isfinite(result[name][select]).all():
                    raise ValueError(
                        f"The output PLY type cannot represent the edited {name} values."
                    )
    return result
