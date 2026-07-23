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
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


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


def geodesic_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Backward-compatible alias for :func:`principal_angle_distance`.

    This historical name measures distance between column spaces, not between
    oriented Stiefel frames. New code should use ``principal_angle_distance``.
    """
    return principal_angle_distance(X, Y)


def cayley_orthogonal(A: np.ndarray) -> np.ndarray:
    """Orthogonal matrix from a skew-symmetric A via the Cayley transform.

    Q = (I - A)^{-1} (I + A) is orthogonal whenever A^T = -A (numpy-only, no scipy).
    """
    n = A.shape[0]
    I = np.eye(n)
    return np.linalg.solve(I - A, I + A)


class RiemannianSGD:
    """
    Reference Riemannian SGD on Stiefel(d, r) (OptimizerProtocol-compatible).

    Reads the current point from ``observation.x`` and the ambient gradient from
    ``observation.grad`` (both flattened), projects the gradient onto the tangent
    space and retracts back to the manifold:
        riem = tangent_project(X, G);  X_next = retract(X, -lr * riem).

    Provided so that the Stiefel landscape can be tracked by a manifold-aware
    optimizer (naive SGD/Adam would leave the manifold).
    """

    name = "RiemannianSGD"
    oracle_type = "first-order"

    def __init__(self, d: int, r: int, lr: float = 0.1):
        if not (1 <= r <= d):
            raise ValueError("Stiefel requires 1 <= r <= d")
        self.d = d
        self.r = r
        self.lr = lr

    def step(self, observation) -> np.ndarray:
        X = observation.x.reshape(self.d, self.r)
        G = observation.grad.reshape(self.d, self.r)
        riem = tangent_project(X, G)
        return retract(X, -self.lr * riem).reshape(-1)

    def reset(self) -> None:
        pass
