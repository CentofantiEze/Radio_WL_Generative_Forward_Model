"""Gaussian Mixture Model utilities for shear posterior density estimation.

Provides GMM fitting, serialization, analytical multiplication, plotting,
and posterior coverage testing for 2D (g1, g2) shear posteriors.
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
    with np.load(filepath) as data:
        return {
            "weights": data["weights"].copy(),
            "means": data["means"].copy(),
            "covariances": data["covariances"].copy(),
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


def gmm_moments(gmm_params):
    """Compute the overall mean and covariance of a GMM.

    Uses the law of total expectation/variance:
        mu = sum_k w_k mu_k
        Sigma = sum_k w_k (Sigma_k + (mu_k - mu)(mu_k - mu)^T)

    Parameters
    ----------
    gmm_params : dict
        GMM parameters.

    Returns
    -------
    mean : ndarray, shape (d,)
    cov : ndarray, shape (d, d)
    """
    weights = gmm_params["weights"]
    means = gmm_params["means"]
    covs = gmm_params["covariances"]

    mu = np.average(means, weights=weights, axis=0)

    # Covariance = weighted sum of (component cov + outer product of deviation)
    d = means.shape[1]
    cov = np.zeros((d, d))
    for k in range(len(weights)):
        diff = means[k] - mu
        cov += weights[k] * (covs[k] + np.outer(diff, diff))

    return mu, cov


def combine_gmms_gaussian(gmm_list, prior_mean=None, prior_cov=None):
    """Combine posteriors by Gaussian moment-matching.

    Extracts the mean and covariance from each GMM (exact mixture moments),
    then combines analytically as Gaussians. This avoids the tail truncation
    problem of direct GMM multiplication.

    The combination is:
        Sigma_combined^{-1} = Sigma_prior^{-1} + sum_i Sigma_Li^{-1}
        mu_combined = Sigma_combined (Sigma_prior^{-1} mu_prior + sum_i Sigma_Li^{-1} mu_Li)

    where Sigma_Li^{-1} = Sigma_i^{-1} - Sigma_prior^{-1} is the likelihood
    precision from run i.

    Parameters
    ----------
    gmm_list : list of dict
        List of GMM parameter dicts.
    prior_mean : array_like, shape (d,), optional
        Mean of the Gaussian prior. Default: zeros.
    prior_cov : array_like, shape (d, d), optional
        Covariance of the Gaussian prior. Required.

    Returns
    -------
    dict
        Single-component GMM dict with the combined posterior.
    """
    if prior_cov is None:
        raise ValueError("prior_cov is required for Gaussian combination")

    prior_cov = np.asarray(prior_cov)
    d = prior_cov.shape[0]
    if prior_mean is None:
        prior_mean = np.zeros(d)
    prior_mean = np.asarray(prior_mean)
    prior_prec = np.linalg.inv(prior_cov)

    import warnings

    # Start with the prior precision
    combined_prec = prior_prec.copy()
    combined_prec_mu = prior_prec @ prior_mean
    n_skipped = 0

    for gmm in gmm_list:
        mu_i, cov_i = gmm_moments(gmm)
        prec_i = np.linalg.inv(cov_i)

        # Likelihood precision = posterior precision - prior precision
        lik_prec = prec_i - prior_prec

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(lik_prec)
        if np.any(eigvals <= 0):
            # Posterior is wider than the prior (uninformative data)
            # Skip — contributes no information
            n_skipped += 1
            continue

        lik_prec_mu = prec_i @ mu_i - prior_prec @ prior_mean

        combined_prec += lik_prec
        combined_prec_mu += lik_prec_mu

    if n_skipped > 0:
        warnings.warn(
            f"combine_gmms_gaussian: {n_skipped}/{len(gmm_list)} posteriors "
            f"wider than the prior (uninformative) — skipped.",
            stacklevel=2,
        )

    combined_cov = np.linalg.inv(combined_prec)
    combined_mean = combined_cov @ combined_prec_mu

    return {
        "weights": np.array([1.0]),
        "means": combined_mean.reshape(1, -1),
        "covariances": combined_cov.reshape(1, d, d),
    }


def divide_gmm_by_gaussian(gmm_params, prior_mean, prior_cov):
    """Divide a GMM by a Gaussian (remove the prior from a posterior).

    For each GMM component N(mu_k, Sigma_k), computes the "likelihood"
    by removing the Gaussian prior:
        Sigma_L = (Sigma_k^{-1} - Sigma_p^{-1})^{-1}
        mu_L = Sigma_L (Sigma_k^{-1} mu_k - Sigma_p^{-1} mu_p)

    This is valid when each component is narrower than the prior.

    Parameters
    ----------
    gmm_params : dict
        GMM parameter dict.
    prior_mean : array_like, shape (d,)
        Mean of the Gaussian prior.
    prior_cov : array_like, shape (d, d)
        Covariance of the Gaussian prior.

    Returns
    -------
    dict
        GMM with prior removed from each component.
    """
    prior_mean = np.asarray(prior_mean)
    prior_cov = np.asarray(prior_cov)
    prior_prec = np.linalg.inv(prior_cov)
    d = len(prior_mean)

    weights = gmm_params["weights"].copy()
    means = gmm_params["means"].copy()
    covs = gmm_params["covariances"].copy()
    K = len(weights)

    import warnings

    new_means = np.zeros_like(means)
    new_covs = np.zeros_like(covs)
    log_weight_corrections = np.zeros(K)
    n_skipped = 0

    for k in range(K):
        prec_k = np.linalg.inv(covs[k])
        prec_new = prec_k - prior_prec

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(prec_new)
        if np.any(eigvals <= 0):
            # Component is wider than the prior — data was uninformative
            # for this component. Set its weight to ~0 so it doesn't
            # contribute (a flat likelihood carries no information).
            n_skipped += 1
            log_weight_corrections[k] = -np.inf
            new_covs[k] = covs[k]
            new_means[k] = means[k]
            continue

        cov_new = np.linalg.inv(prec_new)
        mu_new = cov_new @ (prec_k @ means[k] - prior_prec @ prior_mean)

        new_covs[k] = cov_new
        new_means[k] = mu_new

        # Weight correction: ratio of normalizing constants
        # log c = 0.5 * (log|Sigma_new| - log|Sigma_k| + log|Sigma_p|)
        #       + 0.5 * (mu_new^T Prec_new mu_new - mu_k^T Prec_k mu_k + mu_p^T Prec_p mu_p)
        log_det_new = np.linalg.slogdet(cov_new)[1]
        log_det_k = np.linalg.slogdet(covs[k])[1]
        log_det_p = np.linalg.slogdet(prior_cov)[1]
        log_weight_corrections[k] = 0.5 * (
            log_det_new - log_det_k + log_det_p
            + mu_new @ prec_new @ mu_new
            - means[k] @ prec_k @ means[k]
            + prior_mean @ prior_prec @ prior_mean
        )

    if n_skipped > 0:
        warnings.warn(
            f"divide_gmm_by_gaussian: {n_skipped}/{K} components wider than "
            f"the prior (uninformative) — dropped from likelihood.",
            stacklevel=2,
        )

    # Apply weight corrections
    log_weights = np.log(weights) + log_weight_corrections
    log_weights -= np.max(log_weights)
    new_weights = np.exp(log_weights)
    new_weights /= np.sum(new_weights)

    return {"weights": new_weights, "means": new_means, "covariances": new_covs}


def combine_gmms(gmm_list, weight_threshold=1e-10, max_components=100,
                 prior_mean=None, prior_cov=None):
    """Multiply a list of GMMs sequentially with pruning.

    When prior_mean and prior_cov are provided, divides out (N-1) copies
    of the prior to correct for the fact that each posterior already
    includes the prior. Without this correction, combining N posteriors
    applies the prior N times instead of once, leading to overconfident
    results.

    Parameters
    ----------
    gmm_list : list of dict
        List of GMM parameter dicts to combine.
    weight_threshold : float
        Pruning threshold passed to multiply_gmms.
    max_components : int
        Maximum number of components to keep after each multiplication.
        The lowest-weight components are dropped if exceeded.
    prior_mean : array_like, shape (d,), optional
        Mean of the Gaussian prior on (g1, g2). If provided along with
        prior_cov, divides out (N-1) copies of the prior.
    prior_cov : array_like, shape (d, d), optional
        Covariance of the Gaussian prior on (g1, g2).

    Returns
    -------
    dict
        Combined GMM parameters.
    """
    if len(gmm_list) == 0:
        raise ValueError("gmm_list must be non-empty")
    if len(gmm_list) == 1:
        return gmm_list[0]

    # If prior is given, depriorize each posterior first (posterior / prior = likelihood)
    # then multiply all likelihoods, then multiply by one copy of the prior
    if prior_mean is not None and prior_cov is not None:
        prior_mean = np.asarray(prior_mean)
        prior_cov = np.asarray(prior_cov)
        likelihood_list = [
            divide_gmm_by_gaussian(gmm, prior_mean, prior_cov)
            for gmm in gmm_list
        ]
        # Multiply all likelihoods
        result = likelihood_list[0]
        for lik in likelihood_list[1:]:
            result = multiply_gmms(result, lik, weight_threshold=weight_threshold)
            if len(result["weights"]) > max_components:
                idx = np.argsort(result["weights"])[::-1][:max_components]
                result = {
                    "weights": result["weights"][idx],
                    "means": result["means"][idx],
                    "covariances": result["covariances"][idx],
                }
                result["weights"] /= np.sum(result["weights"])

        # Multiply by one copy of the prior
        prior_gmm = {
            "weights": np.array([1.0]),
            "means": prior_mean.reshape(1, -1),
            "covariances": prior_cov.reshape(1, *prior_cov.shape),
        }
        result = multiply_gmms(result, prior_gmm, weight_threshold=weight_threshold)
        result["weights"] /= np.sum(result["weights"])
        return result

    # No prior correction — original behavior
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


# ---------------------------------------------------------------------------
# Coverage testing
# ---------------------------------------------------------------------------


def credible_level_at_point(gmm_params, point, n_grid=300):
    """Compute the credible level at which a point lies in the GMM posterior.

    Returns the smallest highest-density credible region (as a fraction,
    e.g. 0.68) that contains the given point. A value near 0 means the
    point is at the mode; near 1 means it's deep in the tails.

    Parameters
    ----------
    gmm_params : dict
        GMM parameters.
    point : array_like, shape (2,)
        The point (g1, g2) to test.
    n_grid : int
        Grid resolution per axis. The grid adapts to the GMM extent.

    Returns
    -------
    float
        Credible level in [0, 1].
    """
    point = np.asarray(point).ravel()
    means = gmm_params["means"]
    covs = gmm_params["covariances"]
    weights = gmm_params["weights"]

    # Adaptive grid centered on the GMM, wide enough to cover tails
    stds = np.sqrt(np.abs(covs[:, np.arange(2), np.arange(2)]))
    spread = 5 * np.max(stds, axis=0)
    center = np.average(means, weights=weights, axis=0)

    g1_grid = np.linspace(center[0] - spread[0], center[0] + spread[0], n_grid)
    g2_grid = np.linspace(center[1] - spread[1], center[1] + spread[1], n_grid)
    G1, G2 = np.meshgrid(g1_grid, g2_grid)
    grid_points = np.column_stack([G1.ravel(), G2.ravel()])

    log_density_grid = gmm_log_prob(gmm_params, grid_points)
    log_density_true = gmm_log_prob(gmm_params, point.reshape(1, 2))[0]

    # Credible level = fraction of probability mass at density >= density_true
    density_grid = np.exp(log_density_grid - np.max(log_density_grid))
    density_true = np.exp(log_density_true - np.max(log_density_grid))

    credible_level = np.sum(density_grid[density_grid >= density_true]) / np.sum(
        density_grid
    )
    return float(credible_level)


def compute_coverage(gmm_list, true_g, n_grid=300):
    """Compute credible levels for a list of posteriors at the same true value.

    Parameters
    ----------
    gmm_list : list of dict
        List of GMM parameter dicts, one per independent posterior estimate.
    true_g : array_like, shape (2,)
        True shear value (g1, g2).
    n_grid : int
        Grid resolution for credible level computation.

    Returns
    -------
    ndarray, shape (N,)
        Credible level for each posterior.
    """
    true_g = np.asarray(true_g)
    credible_levels = np.array(
        [credible_level_at_point(gmm, true_g, n_grid=n_grid) for gmm in gmm_list]
    )
    return credible_levels


def coverage_table(credible_levels, nominal_levels=None):
    """Compute empirical coverage at nominal credible levels.

    Parameters
    ----------
    credible_levels : array_like, shape (N,)
        Credible level of the true value in each posterior.
    nominal_levels : list of float, optional
        Nominal levels to evaluate. Default: [0.5, 0.68, 0.8, 0.9, 0.95, 0.99].

    Returns
    -------
    dict
        Keys: 'nominal', 'empirical', 'se', 'n'. Each is an array.
    """
    if nominal_levels is None:
        nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]

    credible_levels = np.asarray(credible_levels)
    N = len(credible_levels)

    empirical = np.array([np.mean(credible_levels <= level) for level in nominal_levels])
    se = np.sqrt(empirical * (1 - empirical) / N)

    return {
        "nominal": np.array(nominal_levels),
        "empirical": empirical,
        "se": se,
        "n": N,
    }


def plot_coverage(credible_levels, ax=None, n_sigma=2, label=None, color="steelblue"):
    """Plot empirical coverage vs nominal confidence level.

    A well-calibrated posterior lies on the 1:1 diagonal. The shaded
    band shows the expected n_sigma binomial uncertainty.

    Parameters
    ----------
    credible_levels : array_like, shape (N,)
        Credible level of the true value in each posterior.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.
    n_sigma : int
        Width of the confidence band in standard deviations.
    label : str, optional
        Label for the curve.
    color : str
        Color of the coverage curve.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    credible_levels = np.asarray(credible_levels)
    N = len(credible_levels)

    # Empirical coverage: for each nominal level p, fraction of runs
    # where credible_level <= p
    nominal = np.linspace(0, 1, 200)
    empirical = np.array([np.mean(credible_levels <= p) for p in nominal])

    # Binomial confidence band around the diagonal
    se = np.sqrt(nominal * (1 - nominal) / N)
    ax.fill_between(
        nominal,
        nominal - n_sigma * se,
        nominal + n_sigma * se,
        alpha=0.15,
        color="gray",
        label=f"${n_sigma}\\sigma$ band (N={N})",
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
    ax.plot(nominal, empirical, color=color, lw=1.5, label=label)

    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()

    return ax
