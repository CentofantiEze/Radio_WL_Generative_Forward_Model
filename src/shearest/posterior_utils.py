"""Gaussian Mixture Model utilities for shear posterior density estimation.

Provides GMM fitting, serialization, analytical multiplication, and plotting
for compact representation of 2D (g1, g2) shear posteriors.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def fit_gmm(samples, n_components=5, random_state=42):
    """Fit a Gaussian Mixture Model to 2D shear samples.

    Parameters
    ----------
    samples : array_like, shape (N, 2)
        Array of (g1, g2) samples.
    n_components : int
        Number of Gaussian components.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        GMM parameters: weights (K,), means (K, 2), covariances (K, 2, 2).
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=300,
    )
    gmm.fit(samples)
    return {
        "weights": gmm.weights_,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
    }


def save_gmm(gmm_params, filepath):
    """Save GMM parameters to a .npz file."""
    np.savez(
        filepath,
        weights=gmm_params["weights"],
        means=gmm_params["means"],
        covariances=gmm_params["covariances"],
    )


def load_gmm(filepath):
    """Load GMM parameters from a .npz file."""
    data = np.load(filepath)
    return {
        "weights": data["weights"],
        "means": data["means"],
        "covariances": data["covariances"],
    }


def gmm_log_prob(gmm_params, points):
    """Evaluate GMM log-density at given points.

    Parameters
    ----------
    gmm_params : dict
        GMM parameters with keys: weights, means, covariances.
    points : array_like, shape (M, 2)
        Points at which to evaluate the density.

    Returns
    -------
    ndarray, shape (M,)
        Log-probability at each point.
    """
    weights = gmm_params["weights"]
    means = gmm_params["means"]
    covariances = gmm_params["covariances"]

    K = len(weights)
    M = len(points)
    log_components = np.zeros((M, K))

    for k in range(K):
        log_components[:, k] = (
            np.log(weights[k])
            + multivariate_normal.logpdf(points, mean=means[k], cov=covariances[k])
        )

    # logsumexp over components
    max_log = np.max(log_components, axis=1, keepdims=True)
    log_prob = max_log[:, 0] + np.log(
        np.sum(np.exp(log_components - max_log), axis=1)
    )
    return log_prob


def multiply_gmms(gmm1, gmm2, weight_threshold=1e-10):
    """Analytically multiply two GMMs.

    The product of two GMM densities is an unnormalized mixture of K1*K2
    Gaussian components. Each pair (i, j) produces a new Gaussian with:
        Sigma_new = (Sigma1_i^{-1} + Sigma2_j^{-1})^{-1}
        mu_new = Sigma_new @ (Sigma1_i^{-1} @ mu1_i + Sigma2_j^{-1} @ mu2_j)
        w_new = w1_i * w2_j * c_ij

    where c_ij is the Gaussian overlap coefficient.

    Parameters
    ----------
    gmm1, gmm2 : dict
        GMM parameter dicts.
    weight_threshold : float
        Drop components with weight below this fraction of the max weight.

    Returns
    -------
    dict
        Combined GMM parameters (unnormalized weights are renormalized).
    """
    w1, m1, c1 = gmm1["weights"], gmm1["means"], gmm1["covariances"]
    w2, m2, c2 = gmm2["weights"], gmm2["means"], gmm2["covariances"]
    K1, K2 = len(w1), len(w2)
    d = m1.shape[1]

    new_weights = []
    new_means = []
    new_covs = []

    for i in range(K1):
        prec1 = np.linalg.inv(c1[i])
        for j in range(K2):
            prec2 = np.linalg.inv(c2[j])

            # Combined precision and covariance
            prec_new = prec1 + prec2
            cov_new = np.linalg.inv(prec_new)

            # Combined mean
            mu_new = cov_new @ (prec1 @ m1[i] + prec2 @ m2[j])

            # Gaussian overlap coefficient
            # c_ij = N(mu1_i; mu2_j, Sigma1_i + Sigma2_j)
            diff = m1[i] - m2[j]
            cov_sum = c1[i] + c2[j]
            log_c = multivariate_normal.logpdf(diff, mean=np.zeros(d), cov=cov_sum)

            log_w = np.log(w1[i]) + np.log(w2[j]) + log_c
            new_weights.append(log_w)
            new_means.append(mu_new)
            new_covs.append(cov_new)

    # Convert from log weights, normalize
    log_weights = np.array(new_weights)
    log_weights -= np.max(log_weights)  # numerical stability
    weights = np.exp(log_weights)

    means = np.array(new_means)
    covs = np.array(new_covs)

    # Prune negligible components
    mask = weights > weight_threshold * np.max(weights)
    weights = weights[mask]
    means = means[mask]
    covs = covs[mask]

    # Renormalize
    weights /= np.sum(weights)

    return {"weights": weights, "means": means, "covariances": covs}


def combine_gmms(gmm_list, weight_threshold=1e-10, max_components=100):
    """Multiply a list of GMMs sequentially with pruning.

    Parameters
    ----------
    gmm_list : list of dict
        List of GMM parameter dicts to combine.
    weight_threshold : float
        Pruning threshold passed to multiply_gmms.
    max_components : int
        Maximum number of components to keep after each multiplication.
        The lowest-weight components are dropped if exceeded.

    Returns
    -------
    dict
        Combined GMM parameters.
    """
    if len(gmm_list) == 0:
        raise ValueError("gmm_list must be non-empty")
    if len(gmm_list) == 1:
        return gmm_list[0]

    result = gmm_list[0]
    for gmm in gmm_list[1:]:
        result = multiply_gmms(result, gmm, weight_threshold=weight_threshold)

        # Cap number of components
        if len(result["weights"]) > max_components:
            idx = np.argsort(result["weights"])[::-1][:max_components]
            result = {
                "weights": result["weights"][idx],
                "means": result["means"][idx],
                "covariances": result["covariances"][idx],
            }
            result["weights"] /= np.sum(result["weights"])

    return result


def plot_gmm_contours(gmm_params, ax=None, levels=(0.68, 0.95), n_grid=200,
                      true_g=None, color="blue", label=None):
    """Plot GMM density contours.

    Parameters
    ----------
    gmm_params : dict
        GMM parameters.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.
    levels : tuple of float
        Probability mass levels for contours.
    n_grid : int
        Grid resolution.
    true_g : tuple of float, optional
        If given, plot a marker at (g1_true, g2_true).
    color : str
        Contour color.
    label : str, optional
        Label for the contour.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    means = gmm_params["means"]
    covs = gmm_params["covariances"]

    # Determine grid range from component means and covariances
    spread = 4 * np.max(np.sqrt(np.abs(covs[:, np.arange(2), np.arange(2)])), axis=0)
    center = np.average(means, weights=gmm_params["weights"], axis=0)
    g1_range = (center[0] - spread[0], center[0] + spread[0])
    g2_range = (center[1] - spread[1], center[1] + spread[1])

    g1_grid = np.linspace(g1_range[0], g1_range[1], n_grid)
    g2_grid = np.linspace(g2_range[0], g2_range[1], n_grid)
    G1, G2 = np.meshgrid(g1_grid, g2_grid)
    points = np.column_stack([G1.ravel(), G2.ravel()])

    log_density = gmm_log_prob(gmm_params, points)
    density = np.exp(log_density).reshape(n_grid, n_grid)

    # Convert probability mass levels to density thresholds
    sorted_density = np.sort(density.ravel())[::-1]
    cumsum = np.cumsum(sorted_density)
    cumsum /= cumsum[-1]
    thresholds = []
    for level in sorted(levels):
        idx = np.searchsorted(cumsum, level)
        if idx < len(sorted_density):
            thresholds.append(sorted_density[idx])
        else:
            thresholds.append(sorted_density[-1])
    # Sort ascending and deduplicate (matplotlib requires strictly increasing levels)
    thresholds = sorted(set(thresholds))

    ax.contour(G1, G2, density, levels=thresholds, colors=color)
    if label is not None:
        ax.plot([], [], color=color, label=label)

    if true_g is not None:
        ax.axvline(true_g[0], color="gray", ls="--", alpha=0.5)
        ax.axhline(true_g[1], color="gray", ls="--", alpha=0.5)

    ax.set_xlabel("g1")
    ax.set_ylabel("g2")
    return ax
