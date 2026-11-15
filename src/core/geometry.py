"""
Stiefel manifold operations for the Riemannian landscape (optional add-on).

Stiefel(d, r) = { X in R^{d x r} : X^T X = I_r } — sets of r orthonormal frames in
R^d. The WIND pipeline stays vector-based: matrices are flattened to length d*r and
reshaped here. These helpers are pure functions of (d, r) and carry no hidden state,
so the landscape, the geodesic metric and a Riemannian optimizer can all share them
without breaking the information barrier (they describe the agent's own search space,
never theta).
"""

import numpy as np


def random_stiefel(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random point on Stiefel(d, r) (d x r, orthonormal columns)."""
    A = rng.normal(size=(d, r))
    Q, R = np.linalg.qr(A, mode="reduced")
    signs = np.where(np.diag(R) < 0, -1.0, 1.0)
    return Q[:, :r] * signs


def project_to_stiefel(M: np.ndarray) -> np.ndarray:
    """Nearest orthonormal frame to M (polar factor via thin SVD)."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def tangent_project(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project an ambient gradient G onto the tangent space at X.

    T_X Stiefel = { Z : X^T Z is skew-symmetric }.  proj_X(G) = G - X sym(X^T G).
    """
    XtG = X.T @ G
    sym = 0.5 * (XtG + XtG.T)
    return G - X @ sym


def retract(X: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Polar retraction of X + xi back onto the manifold."""
    return project_to_stiefel(X + xi)


def principal_angle_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Grassmann geodesic distance between the represented subspaces.

    d(X, Y) = || arccos(sigma_i) ||_2, where sigma_i are the singular values of
    X^T Y (the cosines of the principal angles). Zero iff the frames span the same
    subspace; cheap (one SVD) and well-defined for any orthonormal X, Y.
    """
    s = np.linalg.svd(X.T @ Y, compute_uv=False)
    angles = np.arccos(np.clip(s, -1.0, 1.0))
    return float(np.linalg.norm(angles))


def frame_frobenius_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Extrinsic distance between two oriented Stiefel frames."""
    return float(np.linalg.norm(X - Y, ord="fro"))


def validate_frame(X: np.ndarray, tolerance: float = 1e-6) -> None:
    """Reject invalid representatives instead of silently projecting a metric input."""
    if X.ndim != 2 or X.shape[1] > X.shape[0] or not np.isfinite(X).all():
        raise ValueError("Expected a finite d-by-r frame with r <= d")
    if np.linalg.norm(X.T @ X - np.eye(X.shape[1]), ord="fro") > tolerance:
        raise ValueError("Principal-angle/Stiefel distance requires orthonormal frames")


def canonical_stiefel_distance(X, Y, *, tolerance=1e-8, max_iterations=200):
    """Canonical numerical distance, with no substitution of a different metric.

    Matrix-algebraic logarithm iteration of Zimmermann (2017),
    doi:10.1137/16M1074485. Rank-adaptive completion handles d < 2r.
    Endpoint residual and reverse-direction agreement are necessary numerical
    checks, NOT a certificate of global minimality. Branch-cut/failed solves
    return NaN with diagnostics. Sphere distance is evaluated directly.
    """
    from scipy.linalg import expm, logm, null_space

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError(
            "Positive finite tolerance and positive integer max_iterations required"
        )
    validate_frame(X)
    validate_frame(Y)
    if X.shape != Y.shape:
        raise ValueError("Frame shapes differ")
    d, r = X.shape
    diagnostic = {"metric": "canonical", "globally_certified": False}
    if np.linalg.norm(X - Y) <= tolerance:
        return 0.0, {**diagnostic, "state": "ok", "residual": 0.0, "iterations": 0}
    if d == r and np.linalg.det(X.T @ Y) < 0:
        return float("nan"), {**diagnostic, "state": "disconnected_components"}
    if r == 1:
        value = float(np.arccos(np.clip(float((X.T @ Y)[0, 0]), -1, 1)))
        if abs(value - np.pi) <= tolerance:
            return float("nan"), {**diagnostic, "state": "ambiguous_branch"}
        return value, {
            **diagnostic,
            "state": "ok",
            "residual": 0.0,
            "iterations": 0,
            "closed_form": "sphere",
        }

    def one_way(base, point):
        M = base.T @ point
        U, singular, _ = np.linalg.svd(point - base @ M, full_matrices=False)
        rank = min(d - r, int(np.count_nonzero(singular > 1e-12)))
        Q = U[:, :rank]
        columns = np.vstack((M, Q.T @ point))
        completion = null_space(columns.T)
        V = np.column_stack((columns, completion))
        if rank:
            u, _, vt = np.linalg.svd(V[r:, r:])
            V[:, r:] = V[:, r:] @ vt.T @ u.T
            if np.linalg.det(V) < 0:
                V[:, -1] *= -1
        basis = np.column_stack((base, Q))
        for iteration in range(1, max_iterations + 1):
            if np.min(np.abs(np.linalg.eigvals(V) + 1)) < 1e-8:
                return None, {"state": "ambiguous_branch", "iterations": iteration}
            logarithm = logm(V)
            if (
                not np.isfinite(logarithm).all()
                or np.max(np.abs(np.imag(logarithm))) > tolerance
            ):
                return None, {"state": "complex_logarithm", "iterations": iteration}
            logarithm = np.real(logarithm)
            logarithm = 0.5 * (logarithm - logarithm.T)
            C = logarithm[r:, r:]
            if np.linalg.norm(C, ord="fro") <= tolerance:
                A, B = logarithm[:r, :r], logarithm[r:, :r]
                generator = np.block([[A, -B.T], [B, np.zeros((rank, rank))]])
                endpoint = basis @ expm(generator)[:, :r]
                residual = float(np.linalg.norm(endpoint - point, ord="fro"))
                if residual > 10 * tolerance:
                    return None, {"state": "endpoint_residual", "residual": residual}
                norm = float(np.sqrt(0.5 * np.sum(A * A) + np.sum(B * B)))
                return norm, {
                    "state": "ok",
                    "iterations": iteration,
                    "residual": residual,
                }
            V[:, r:] = V[:, r:] @ expm(-C)
        return None, {"state": "not_converged", "iterations": max_iterations}

    try:
        forward, f_info = one_way(X, Y)
        reverse, r_info = one_way(Y, X)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return float("nan"), {
            **diagnostic,
            "state": "numerical_failure",
            "reason": str(exc),
        }
    if forward is None or reverse is None:
        return float("nan"), {
            **diagnostic,
            "state": "unavailable",
            "forward": f_info,
            "reverse": r_info,
        }
    disagreement = abs(forward - reverse)
    if disagreement > 20 * tolerance * max(1, forward, reverse):
        return float("nan"), {
            **diagnostic,
            "state": "branch_disagreement",
            "symmetry_residual": disagreement,
        }
    return 0.5 * (forward + reverse), {
        **diagnostic,
        "state": "ok",
        "forward": f_info,
        "reverse": r_info,
        "symmetry_residual": disagreement,
    }


def cayley_orthogonal(A: np.ndarray) -> np.ndarray:
    """Orthogonal matrix from a skew-symmetric A via the Cayley transform.

    Q = (I - A)^{-1} (I + A) is orthogonal whenever A^T = -A (numpy-only, no scipy).
    """
    n = A.shape[0]
    I = np.eye(n)
    return np.linalg.solve(I - A, I + A)
