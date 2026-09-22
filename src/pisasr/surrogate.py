"""Gaussian-process (GP) criticality surrogate (scikit-learn).

The surrogate maps scenario features to the recourse cost under the reference
commitment. Inputs are standardized, the target is standardized with its mean
and population standard deviation (``numpy.std``), and the kernel is a constant
times an anisotropic (automatic relevance determination) Matérn-5/2 kernel plus a
white-noise term. Hyperparameters are fitted by maximizing the log marginal
likelihood with four optimizer restarts.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class GPSurrogate:
    """GP regression with standardized inputs and target.

    Parameters
    ----------
    length_scale_dim
        Number of features; one length scale is fitted for each feature. With
        ``None`` a single (isotropic) length scale is used.
    alpha
        Value added to the diagonal of the kernel matrix.
    n_restarts
        Restarts of the hyperparameter optimizer.
    random_state
        Random seed of the optimizer restarts.
    """

    def __init__(self, length_scale_dim: int | None = None, alpha: float = 1e-6,
                 n_restarts: int = 4, random_state: int = 0):
        ls0 = np.ones(length_scale_dim) if length_scale_dim else 1.0
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e6))
            * Matern(length_scale=ls0, length_scale_bounds=(1e-3, 1e6), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e2))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
        )
        self.x_scaler = StandardScaler()
        self.y_mean = 0.0
        self.y_std = 1.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        """Fit the GP to features ``X`` and recourse costs ``y``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xs = self.x_scaler.fit_transform(X)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        ys = (y - self.y_mean) / self.y_std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.gp.fit(Xs, ys)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Posterior mean (and standard deviation) in the original cost units."""
        Xs = self.x_scaler.transform(np.asarray(X, dtype=float))
        if return_std:
            mu, sd = self.gp.predict(Xs, return_std=True)
            return mu * self.y_std + self.y_mean, sd * self.y_std
        mu = self.gp.predict(Xs)
        return mu * self.y_std + self.y_mean

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Coefficient of determination (R^2) and mean absolute error on ``(X, y)``."""
        pred = self.predict(X)
        y = np.asarray(y, dtype=float).ravel()
        return {
            "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
        }
