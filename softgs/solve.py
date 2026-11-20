"""Source quadrature and hard-control response."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import LinAlgWarning, eigh, solve
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu
import warnings

from .basis import design


def affine_coefficients(atlas):
    a = np.zeros((len(atlas["centers"]), 16, 2, 4))
    a[:, :, 0, 0] = 1.0
    a[:, :, 0, 1:] = atlas["controls"].reshape(-1, 16, 3)
    a[:, :, 1, 1:] = atlas["radii"][:, None, None] * atlas["axes"][:, 2, None, :]
    return a.reshape(-1, 4)


def gaussian_measure(positions, scales, opacity, source, spans):
    """Accumulate opacity-weighted Gaussian support area in a source grid."""
    points = np.concatenate((positions, source))
    lower = points.min(axis=0)
    width = np.ptp(points, axis=0)
    if np.any(width <= 0):
        raise ValueError(
            "The source bounding box must have positive extent on each axis."
        )
    coordinates = (positions - lower) / width
    cells = np.minimum((coordinates * spans).astype(int), spans - 1)
    indices = np.ravel_multi_index(cells.T, spans)
    # Normalize in log space before summing areas of the principal planes.
    log_area = np.logaddexp.reduce(scales.sum(axis=1, keepdims=True) - scales, axis=1)
    log_mass = log_area - np.logaddexp(0.0, -opacity)
    weights = np.exp(log_mass - log_mass.max())
    density = np.bincount(indices, weights=weights, minlength=np.prod(spans))
    density = 0.01 + 0.99 * len(density) * density / density.sum()
    return lower, width, spans, density


def quadrature(lower, width, spans, density):
    nodes, weights = leggauss(4)
    offsets = np.indices((4, 4, 4)).reshape(3, -1).T
    nodes = ((nodes + 1) / 2)[offsets]
    weights = np.prod((weights / 2)[offsets], axis=1)
    cells = np.indices(tuple(spans)).reshape(3, -1).T
    cell_width = width / spans
    sites = lower + (cells[:, None] + nodes) * cell_width
    volume_weights = np.tile(weights, len(cells)) * np.prod(cell_width)
    return sites.reshape(-1, 3), volume_weights, np.repeat(density, len(nodes))


def integrate(atlas, sites, volume, density, batch_size=256, gram_device="cpu"):
    """Integrate the source bending and mass matrices."""
    size = 32 * len(atlas["centers"])
    bending = np.zeros((size, size))
    mass = np.zeros_like(bending)
    if gram_device != "cpu":
        import torch

    def gram(basis, weight):
        active = np.unique(basis.indices)
        dense = basis[:, active].toarray()
        if gram_device != "cpu" and len(active) >= 512:
            matrix = torch.from_numpy(dense).to(gram_device)
            weights = torch.from_numpy(weight).to(gram_device)
            return np.ix_(active, active), (
                matrix.T @ (weights[:, None] * matrix)
            ).cpu().numpy()
        return np.ix_(active, active), dense.T @ (weight[:, None] * dense)

    for start in range(0, len(sites), batch_size):
        block = slice(start, start + batch_size)
        value, _, hessian = design(atlas, sites[block], second=True)
        indices, values = gram(hessian, np.tile(volume[block] * density[block], 6))
        bending[indices] += values
        indices, values = gram(value, volume[block])
        mass[indices] += values
    return bending, mass


def hard_control_response(energy, trace):
    """Solve the scaled KKT system with one right-hand side per control."""
    scale = np.sqrt(np.maximum(np.diag(energy), 0))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("The source bending energy has an unobservable coefficient.")
    inverse = diags(1 / scale, format="csc")
    scaled_energy = inverse @ csc_matrix(energy) @ inverse
    trace = csc_matrix(trace)
    constraint = trace @ inverse
    row_scale = np.sqrt(np.asarray(constraint.power(2).sum(axis=1)).ravel())
    if np.any(row_scale <= 0):
        raise ValueError("A control lies outside the source chart support.")
    constraint = diags(1 / row_scale) @ constraint
    # Adding C.T @ C to the factorization leaves the constrained optimum unchanged.
    try:
        augmented = (scaled_energy + constraint.T @ constraint).tocsc()
        factor = splu(
            augmented,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        cross = factor.solve(constraint.T.toarray())
        schur = constraint @ cross
        values = np.diag(1 / row_scale)
        coefficient = np.zeros_like(cross)
        dual = np.zeros_like(values)
        residual_values = values
        residual_force = np.zeros_like(cross)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            for _ in range(4):
                unconstrained = factor.solve(residual_force)
                multipliers = solve(schur, residual_values - constraint @ unconstrained)
                coefficient += unconstrained + cross @ multipliers
                dual += residual_values - multipliers
                residual_values = values - constraint @ coefficient
                residual_force = -(scaled_energy @ coefficient + constraint.T @ dual)
                if np.max(np.abs(residual_values)) < 1e-9 and np.linalg.norm(
                    residual_force
                ) < (1e-9 * max(np.linalg.norm(scaled_energy @ coefficient), 1e-20)):
                    break
        response = coefficient / scale[:, None]
        error = np.max(np.abs(trace @ response - np.eye(len(row_scale))))
        relative = np.linalg.norm(residual_force) / max(
            np.linalg.norm(scaled_energy @ coefficient), 1e-20
        )
        if np.isfinite(response).all() and max(error, relative) <= 1e-5:
            return response
    except (RuntimeError, np.linalg.LinAlgError, LinAlgWarning):
        pass  # If the factorization is ill-conditioned, solve the original KKT system.
    system = bmat([[scaled_energy, constraint.T], [constraint, None]], format="csc")
    rhs = np.vstack((np.zeros((len(scale), len(row_scale))), np.diag(1 / row_scale)))
    try:
        factor = splu(system, permc_spec="MMD_AT_PLUS_A")
    except RuntimeError as exc:
        raise ValueError(
            "The source controls do not define an independent constrained solve."
        ) from exc
    solution = factor.solve(rhs)
    for _ in range(3):
        residual = rhs - system @ solution
        if np.max(np.abs(residual)) < 1e-9:
            break
        solution += factor.solve(residual)
    return solution[: len(scale)] / scale[:, None]


def prepare_response(atlas, source, mass, bending, alias_cutoff=1e-11):
    """Remove numerical aliases in the physical L2 inner product."""
    affine_input = np.column_stack((np.ones(len(source)), source))
    if np.linalg.matrix_rank(affine_input) != 4:
        raise ValueError("Source controls must span 3D to constrain the affine modes.")
    trace, _ = design(atlas, source)
    active = np.diag(mass) > 0
    reduced_mass = mass[np.ix_(active, active)]
    scale = np.sqrt(np.diag(reduced_mass))
    scaled = reduced_mass / scale[:, None] / scale[None, :]
    eigenvalues, vectors = eigh((scaled + scaled.T) / 2)
    keep = eigenvalues > alias_cutoff * eigenvalues[-1]
    if np.count_nonzero(keep) < len(source):
        raise ValueError(
            "The source chart space has fewer independent modes than controls."
        )
    transform = np.zeros((len(mass), np.count_nonzero(keep)))
    transform[active] = (vectors[:, keep] / np.sqrt(eigenvalues[keep])) / scale[:, None]
    energy = transform.T @ bending @ transform
    reduced_trace = trace @ transform
    response = transform @ hard_control_response((energy + energy.T) / 2, reduced_trace)
    # Complete the affine response in the full chart space.
    modes = affine_coefficients(atlas)
    response += (modes - response @ affine_input) @ np.linalg.pinv(affine_input)
    error = np.max(np.abs(trace @ response - np.eye(len(source))))
    if not np.isfinite(response).all() or error > 1e-5:
        raise ValueError(f"Source response violates hard controls (error {error:.3g}).")
    return response
