"""Bicubic chart basis and derivatives in canonical coordinates."""

import numpy as np
from scipy.sparse import coo_matrix


def bernstein(t):
    return np.stack(
        [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3], axis=-1
    )


def d_bernstein(t):
    return np.stack(
        [-3 * (1 - t) ** 2, 3 * (1 - t) * (1 - 3 * t), 3 * t * (2 - 3 * t), 3 * t**2],
        axis=-1,
    )


def d2_bernstein(t):
    return np.stack((6 * (1 - t), -12 + 18 * t, 6 - 18 * t, 6 * t), axis=-1)


def bezier_jet(control, uv, second=False):
    """Evaluate a tensor-product Bernstein patch and its parameter derivatives."""
    u, v = uv.T
    bu, bv, du, dv = bernstein(u), bernstein(v), d_bernstein(u), d_bernstein(v)

    def evaluate(a, b):
        return np.einsum("na,nab...,nb->n...", a, control, b, optimize=True)

    value = evaluate(bu, bv)
    derivative = np.stack((evaluate(du, bv), evaluate(bu, dv)), axis=-1)
    if not second:
        return value, derivative, None
    uu, uvv, vv = (
        evaluate(d2_bernstein(u), bv),
        evaluate(du, dv),
        evaluate(bu, d2_bernstein(v)),
    )
    hessian = np.stack(
        (np.stack((uu, uvv), axis=-1), np.stack((uvv, vv), axis=-1)), axis=-2
    )
    return value, derivative, hessian


def fixed_overlap_kernel(points, center, radius, second=False):
    """C2 source support, with exact first/second physical derivatives."""
    relative = points - center
    q = np.minimum(np.linalg.norm(relative, axis=1) / radius, 1.0)
    value = (1 - q) ** 4 * (4 * q + 1)
    gradient = (-20 * (1 - q) ** 3 / radius**2)[:, None] * relative
    hessian = None
    if second:
        hessian = (-20 * (1 - q) ** 3 / radius**2)[:, None, None] * np.eye(3) + (
            60 * (1 - q) ** 2 / (radius**4 * np.maximum(q, 1e-15))
        )[:, None, None] * relative[:, :, None] * relative[:, None]
    return value, gradient, hessian


def design(atlas, points, second=False):
    """Sparse PU value/gradient/Hessian, including source-height derivatives."""
    n = len(points)
    count = len(atlas["centers"])
    size = 32 * count
    kernels = []
    gradients = []
    hessians = []
    for center, radius in zip(atlas["centers"], atlas["radii"]):
        v, g, h = fixed_overlap_kernel(points, center, radius, second)
        kernels.append(v)
        gradients.append(g)
        if second:
            hessians.append(h)
    kernels = np.stack(kernels, 1)
    kg = np.stack(gradients, 1)
    total = kernels.sum(1)
    if np.any(total <= 1e-14):
        raise ValueError("Source query outside the fixed surface cover")
    w = kernels / total[:, None]
    tg = kg.sum(1)
    dw = (kg - w[..., None] * tg[:, None]) / total[:, None, None]
    if second:
        kh = np.stack(hessians, 1)
        wh = (
            kh
            - w[..., None, None] * kh.sum(1)[:, None]
            - dw[..., :, None] * tg[:, None, None, :]
            - tg[:, None, :, None] * dw[..., None, :]
        ) / total[:, None, None, None]
    rows = []
    cols = []
    values = []
    dvalues = []
    hvalues = []
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def tensor(u, v, normal):
        return np.einsum("na,nb,nm->nabm", u, v, normal).reshape(-1, 32)

    for k in range(count):
        sel = np.flatnonzero(kernels[:, k] > 0)
        if not len(sel):
            continue
        axis = atlas["axes"][k]
        radius = atlas["radii"][k]
        center = atlas["centers"][k]
        uv = ((points[sel] - center) @ axis[:2].T / radius + 1) / 2
        pi = axis[:2] / (2 * radius)
        source, ds, d2s = bezier_jet(
            np.broadcast_to(atlas["controls"][k], (len(sel), 4, 4, 3)), uv, second
        )
        normal = axis[2] / radius
        rho = (points[sel] - source) @ normal
        dr = normal - np.einsum("i,nip,pj->nj", normal, ds, pi)
        bu, bv = bernstein(uv[:, 0]), bernstein(uv[:, 1])
        du, dv = d_bernstein(uv[:, 0]), d_bernstein(uv[:, 1])
        power = np.column_stack((np.ones(len(sel)), rho))
        first = np.column_stack((np.zeros(len(sel)), np.ones(len(sel))))
        b = tensor(bu, bv, power)
        dbr = tensor(bu, bv, first)
        db = (
            tensor(du, bv, power)[..., None] * pi[0]
            + tensor(bu, dv, power)[..., None] * pi[1]
            + dbr[..., None] * dr[:, None]
        )
        if second:
            param_h = np.zeros((len(sel), 32, 3, 3))
            param_h[:, :, 0, 0] = tensor(d2_bernstein(uv[:, 0]), bv, power)
            param_h[:, :, 1, 1] = tensor(bu, d2_bernstein(uv[:, 1]), power)
            param_h[:, :, 0, 1] = param_h[:, :, 1, 0] = tensor(du, dv, power)
            param_h[:, :, 0, 2] = param_h[:, :, 2, 0] = tensor(du, bv, first)
            param_h[:, :, 1, 2] = param_h[:, :, 2, 1] = tensor(bu, dv, first)
            dq = np.broadcast_to(pi, (len(sel), 2, 3))
            dq = np.concatenate((dq, dr[:, None]), 1)
            rho_h = -np.einsum("i,nipq,pa,qb->nab", normal, d2s, pi, pi)
            bh = (
                np.einsum("nfab,nai,nbj->nfij", param_h, dq, dq, optimize=True)
                + dbr[..., None, None] * rho_h[:, None]
            )
            blend = (
                w[sel, k, None, None, None] * bh
                + db[..., :, None] * dw[sel, k, None, None, :]
                + dw[sel, k, None, :, None] * db[..., None, :]
                + b[..., None, None] * wh[sel, k, None]
            )
            hvalues.append(
                np.stack(
                    [
                        (1 if a == v else np.sqrt(2)) * blend[:, :, a, v]
                        for a, v in pairs
                    ]
                )
            )
        db = w[sel, k, None, None] * db + b[..., None] * dw[sel, k, None]
        rows.append(np.repeat(sel, 32))
        cols.append(np.tile(np.arange(k * 32, (k + 1) * 32), len(sel)))
        values.append((w[sel, k, None] * b).ravel())
        dvalues.append(db.transpose(2, 0, 1).reshape(3, -1))
    rr = np.concatenate(rows)
    cc = np.concatenate(cols)
    vv = np.concatenate(values)
    value = coo_matrix((vv, (rr, cc)), shape=(n, size)).tocsr()
    derivative = coo_matrix(
        (
            np.concatenate(dvalues, 1).ravel(),
            (np.concatenate([rr + a * n for a in range(3)]), np.tile(cc, 3)),
        ),
        shape=(3 * n, size),
    ).tocsr()
    if not second:
        return value, derivative
    hessian = coo_matrix(
        (
            np.concatenate([v.reshape(6, -1) for v in hvalues], 1).ravel(),
            (np.concatenate([rr + a * n for a in range(6)]), np.tile(cc, 6)),
        ),
        shape=(6 * n, size),
    ).tocsr()
    return value, derivative, hessian


def evaluate(atlas, coefficient, points, batch=1024):
    def tensor(u, v):
        return np.einsum("na,nb->nab", u, v).reshape(-1, 16)

    value = []
    derivative = []
    coefficients = coefficient.reshape(-1, 16, 2, 3)
    for start in range(0, len(points), batch):
        x = points[start : start + batch]
        n = len(x)
        total = np.zeros(n)
        total_g = np.zeros_like(x)
        numerator = np.zeros_like(x)
        numerator_g = np.zeros((n, 3, 3))
        # Contract the actual surface/director controls directly. Avoid building
        # the large source response design matrix for one edited coefficient state.
        for k, (center, radius) in enumerate(zip(atlas["centers"], atlas["radii"])):
            kernel, kg, _ = fixed_overlap_kernel(x, center, radius)
            total += kernel
            total_g += kg
            sel = np.flatnonzero(kernel > 0)
            if not len(sel):
                continue
            axis = atlas["axes"][k]
            relative = x[sel] - center
            uv = (relative @ axis[:2].T / radius + 1) / 2
            pi = axis[:2] / (2 * radius)
            bu, bv = bernstein(uv[:, 0]), bernstein(uv[:, 1])
            du, dv = d_bernstein(uv[:, 0]), d_bernstein(uv[:, 1])
            basis = tensor(bu, bv)
            dbu = tensor(du, bv)
            dbv = tensor(bu, dv)
            height = (atlas["controls"][k].reshape(16, 3) - center) @ axis[2]
            rho = (relative @ axis[2] - basis @ height) / radius
            dr = (
                axis[2] / radius
                - ((dbu @ height)[:, None] * pi[0] + (dbv @ height)[:, None] * pi[1])
                / radius
            )
            uv_fields = np.einsum("na,amj->nmj", basis, coefficients[k])
            first_u = np.einsum("na,amj->nmj", dbu, coefficients[k])
            first_v = np.einsum("na,amj->nmj", dbv, coefficients[k])
            displacement = uv_fields[:, 0] + rho[:, None] * uv_fields[:, 1]
            gradient = (
                (first_u[:, 0] + rho[:, None] * first_u[:, 1])[:, :, None] * pi[0]
                + (first_v[:, 0] + rho[:, None] * first_v[:, 1])[:, :, None] * pi[1]
                + uv_fields[:, 1, :, None] * dr[:, None, :]
            )
            numerator[sel] += kernel[sel, None] * displacement
            numerator_g[sel] += (
                kernel[sel, None, None] * gradient
                + displacement[:, :, None] * kg[sel, None, :]
            )
        if np.any(total <= 1e-14):
            raise ValueError("Source evaluation outside fixed surface cover")
        displacement = numerator / total[:, None]
        value.append(x + displacement)
        derivative.append(
            np.eye(3)
            + (numerator_g - displacement[:, :, None] * total_g[:, None, :])
            / total[:, None, None]
        )
    return np.concatenate(value), np.concatenate(derivative)
