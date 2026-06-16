"""
Uncertainty Metrics for Bayesian Selective Prediction

Per-class epistemic uncertainty (C_k) and associated metrics
from the 2nd-order Taylor expansion of mutual information.

=== PROPOSED METHODS (C_k-based) ===
    C_k(x) = (1/2) * Var[p_k] / mu_k       per-class epistemic uncertainty
    rho_k(x) = |m_{3,k}| / (3*mu_k*Var[p_k])  skewness reliability diagnostic
    C_critical_sum = sum_{k in K_crit} C_k   total epistemic on critical classes
    C_critical_max = max_{k in K_crit} C_k   max-epistemic critical class

=== STRUCTURE-AWARE PROPOSED (C_k + covariance) ===
    CBEC:           sum_{i in S, j in C} sqrt(C_i * C_j) * max(0, -rho_{ij})

=== SCALAR BASELINES ===
    Entropy:   H[mu] = -sum_k mu_k log(mu_k)
    MI:        H[mu] - (1/S) sum_s H[p^(s)]
    MaxProb:   1 - max_k(mu_k)

=== PER-CLASS BASELINES ===
    Var_critical:       max_{k in K_crit} Var[p_k]
    Sale_EU_k:          Var[p_k] (variance-based label-wise EU, Sale et al.)
    Sale_EU_critical:   sum_{k in K_crit} Var[p_k]  (Sale global on critical)
    Sale_EU_global:     sum_k Var[p_k]  (Sale global over all classes)
    OneVsAll_MI_k:      per-class MI via binary reformulation

=== DIAGNOSTIC ANALYSIS (C_k interpretability) ===
    Epistemic profiles:     E[C_k | y=i] -- which classes drive uncertainty per true label
    Error signatures:       E[C_k | y=i, yhat=j] -- fingerprint each error type
    Epistemic dominance:    argmax_k C_k, fraction, entropy -- focused vs diffuse uncertainty
    Confusion matrix:       E[sqrt(C_i*C_j)*max(0,-rho_ij)] -- pairwise epistemic confusion
    Drift monitoring:       E_t[C_k] over batches -- per-class epistemic shift detection

Reference: per_class_epistemic_uncertainty_v2_G-3.pdf

Usage:
    mc_preds, y_true = run_mc_inference(model, dataset, n_samples=100)
    metrics = compute_all_uncertainties(mc_preds, critical_classes=[3])
    policies = build_deferral_policies(metrics)
"""

import numpy as np


# =============================================================================
# Core Predictive Statistics
# =============================================================================

def compute_predictive_statistics(mc_predictions):
    """
    Compute predictive mean, variance, and third central moment
    from Monte Carlo predictions.

    Args:
        mc_predictions: np.ndarray [S, N, K]
            S stochastic forward passes, N samples, K classes.

    Returns:
        mu:  np.ndarray [N, K] -- predictive mean per class.
        var: np.ndarray [N, K] -- predictive variance per class.
        m3:  np.ndarray [N, K] -- third central moment per class.
    """
    mu = np.mean(mc_predictions, axis=0)                          # [N, K]
    var = np.var(mc_predictions, axis=0)                          # [N, K]
    m3 = np.mean((mc_predictions - mu[np.newaxis, :, :]) ** 3,
                 axis=0)                                          # [N, K]
    return mu, var, m3


# =============================================================================
# PROPOSED: Per-Class Epistemic Uncertainty (C_k)
# =============================================================================

def compute_per_class_epistemic(mu, var, eps=1e-10):
    """
    Per-class epistemic uncertainty (proposed metric).

        C_k(x) = (1/2) * Var[p_k] / mu_k

    From the 2nd-order Taylor expansion of mutual information.
    Each C_k measures how much weight uncertainty contributes to
    predictive uncertainty for class k. The 1/mu_k factor corrects
    the boundary suppression that raw variance exhibits near p_k ~ 0.

    Args:
        mu:  np.ndarray [N, K] -- predictive mean.
        var: np.ndarray [N, K] -- predictive variance.
        eps: Small constant to avoid division by zero.

    Returns:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
    """
    return 0.5 * var / (mu + eps)


def compute_skewness_diagnostic(mu, var, m3, eps=1e-10):
    """
    Skewness reliability diagnostic for C_k.

        rho_k(x) = |m_{3,k}| / (3 * mu_k * Var[p_k])

    Ratio of the 3rd-order Taylor correction to the 2nd-order term.
    When rho_k << 1, the quadratic approximation is reliable and
    C_k can be trusted. When rho_k > 0.3, the second-order
    approximation is degrading and C_k should be interpreted
    with caution.

    Derivation: The 2nd-order MI contribution from class k is
    (1/2)*Var[p_k]/mu_k. The 3rd-order correction is
    (1/6)*|m_{3,k}|/mu_k^2. Their ratio gives rho_k.

    Args:
        mu:  np.ndarray [N, K] -- predictive mean.
        var: np.ndarray [N, K] -- predictive variance.
        m3:  np.ndarray [N, K] -- third central moment.
        eps: Small constant to avoid division by zero.

    Returns:
        rho_k: np.ndarray [N, K] -- per-class reliability indicator.
    """
    return np.abs(m3) / (3.0 * (mu + eps) * (var + eps))


def compute_critical_epistemic_sum(C_k, critical_classes):
    """
    Total epistemic on critical classes (proposed deferral score).

        C_crit(x) = sum_{k in K_crit} C_k(x)

    Defer if C_crit(x) > tau.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        critical_classes: list of int -- indices of critical classes.

    Returns:
        np.ndarray [N] -- critical epistemic sum per sample.
    """
    return np.sum(C_k[:, critical_classes], axis=1)


def compute_critical_epistemic_max(C_k, critical_classes):
    """
    Max-epistemic critical class (proposed deferral score).

        C_max(x) = max_{k in K_crit} C_k(x)

    Defer if C_max(x) > tau.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        critical_classes: list of int -- indices of critical classes.

    Returns:
        np.ndarray [N] -- critical epistemic max per sample.
    """
    return np.max(C_k[:, critical_classes], axis=1)


# =============================================================================
# SCALAR BASELINES: Entropy, MI, MaxProb
# =============================================================================

def compute_predictive_entropy(mu, eps=1e-10):
    """
    Predictive entropy of the mean prediction.

        H[mu] = -sum_k mu_k * log(mu_k)

    Measures total predictive uncertainty (epistemic + aleatoric).

    Args:
        mu:  np.ndarray [N, K] -- predictive mean.
        eps: Small constant for numerical stability.

    Returns:
        entropy: np.ndarray [N] -- predictive entropy per sample.
    """
    return -np.sum(mu * np.log(mu + eps), axis=1)


def compute_mutual_information(mc_predictions, mu=None, eps=1e-10):
    """
    Mutual information between predictions and model parameters.

        MI = H[mu] - (1/S) * sum_s H[p^(s)]

    Measures epistemic uncertainty (reducible with more data).
    Standard scalar baseline for selective prediction.

    Args:
        mc_predictions: np.ndarray [S, N, K] -- MC samples.
        mu:  np.ndarray [N, K] or None -- precomputed mean (optional).
        eps: Small constant for numerical stability.

    Returns:
        mi: np.ndarray [N] -- mutual information per sample.
    """
    if mu is None:
        mu = np.mean(mc_predictions, axis=0)

    H_mean = -np.sum(mu * np.log(mu + eps), axis=1)            # [N]

    H_individual = -np.sum(
        mc_predictions * np.log(mc_predictions + eps), axis=2
    )                                                           # [S, N]
    H_expected = np.mean(H_individual, axis=0)                  # [N]

    return H_mean - H_expected


def compute_maxprob_uncertainty(mu):
    """
    MaxProb uncertainty: 1 - max_k(mu_k).

    Simple confidence-based scalar baseline. Higher = less confident.

    Args:
        mu: np.ndarray [N, K] -- predictive mean.

    Returns:
        unc: np.ndarray [N] -- MaxProb uncertainty per sample.
    """
    return 1.0 - np.max(mu, axis=1)


# =============================================================================
# PER-CLASS BASELINES: Variance thresholding, Sale et al., One-vs-All MI
# =============================================================================

def compute_critical_variance_max(var, critical_classes):
    """
    Per-class variance thresholding (per-class baseline).

    Defer if any Var[p_k] for k in K_crit exceeds threshold.

        score(x) = max_{k in K_crit} Var[p_k](x)

    Args:
        var: np.ndarray [N, K] -- predictive variance.
        critical_classes: list of int -- indices of critical classes.

    Returns:
        np.ndarray [N] -- max variance for critical classes.
    """
    return np.max(var[:, critical_classes], axis=1)


def compute_sale_eu(var):
    """
    Sale et al. variance-based label-wise epistemic uncertainty.

        EU^Sale_k(x) = Var[Theta_k(x)]  (under squared loss)

    Under MC approximation with squared loss, this is numerically
    equivalent to Var[p_k]. We label it separately to distinguish
    the axiomatic variance-EU framework from MI-based C_k.

    Sale et al. define label-wise EU via proper scoring rules.
    Under squared loss (Brier score), the epistemic uncertainty
    functional for class k reduces to Var[Theta_k], where
    Theta_k is the random true class probability. Approximated
    via MC samples as Var[p_k^(s)].

    Args:
        var: np.ndarray [N, K] -- predictive variance = Var[p_k].

    Returns:
        EU_sale: np.ndarray [N, K] -- label-wise EU (= var).
    """
    # Numerically identical to var; kept as separate function for
    # conceptual clarity and proper attribution in experiments.
    return var.copy()


def compute_sale_eu_global(var):
    """
    Sale et al. global EU: sum over all classes.

        EU^Sale_global(x) = sum_k Var[p_k](x)

    Defines a global epistemic uncertainty functional distinct from MI.

    Args:
        var: np.ndarray [N, K] -- predictive variance.

    Returns:
        np.ndarray [N] -- global variance-EU per sample.
    """
    return np.sum(var, axis=1)


def compute_sale_eu_critical(var, critical_classes):
    """
    Sale et al. EU restricted to critical classes.

        EU^Sale_crit(x) = sum_{k in K_crit} Var[p_k](x)

    Per-class baseline analogous to C_critical_sum but using
    raw variance instead of C_k = Var/(2*mu).

    Args:
        var: np.ndarray [N, K] -- predictive variance.
        critical_classes: list of int.

    Returns:
        np.ndarray [N] -- Sale EU on critical classes.
    """
    return np.sum(var[:, critical_classes], axis=1)


def compute_one_vs_all_mi(mc_predictions, critical_classes, eps=1e-10):
    """
    One-vs-All MI for critical classes.

    Reformulate the K-class problem as a binary problem per
    critical class: "class k" vs "not class k". Then compute
    standard MI on the binary distribution.

    For each critical class k:
        q_k^(s) = p_k^(s)          (probability of class k)
        q_not_k^(s) = 1 - p_k^(s)  (probability of not class k)

        MI_k = H[mean binary] - mean_s H[binary^(s)]

    Deferral score = sum of MI_k over critical classes.

    Args:
        mc_predictions: np.ndarray [S, N, K] -- MC samples.
        critical_classes: list of int.
        eps: Small constant for numerical stability.

    Returns:
        ova_mi_per_class: np.ndarray [N, len(critical_classes)]
            -- one-vs-all MI for each critical class.
        ova_mi_sum: np.ndarray [N]
            -- sum over critical classes (deferral score).
    """
    S, N, K = mc_predictions.shape
    ova_mi_list = []

    for k in critical_classes:
        # Binary probabilities: [S, N, 2]
        q_k = mc_predictions[:, :, k]            # [S, N]
        q_not_k = 1.0 - q_k                      # [S, N]

        # Mean binary distribution
        mu_k = np.mean(q_k, axis=0)              # [N]
        mu_not_k = 1.0 - mu_k                    # [N]

        # Entropy of mean binary distribution
        H_mean = -(mu_k * np.log(mu_k + eps)
                   + mu_not_k * np.log(mu_not_k + eps))       # [N]

        # Mean entropy of individual binary samples
        H_ind = -(q_k * np.log(q_k + eps)
                  + q_not_k * np.log(q_not_k + eps))          # [S, N]
        H_expected = np.mean(H_ind, axis=0)                    # [N]

        mi_k = H_mean - H_expected                             # [N]
        ova_mi_list.append(mi_k)

    ova_mi_per_class = np.stack(ova_mi_list, axis=1)  # [N, len(crit)]
    ova_mi_sum = np.sum(ova_mi_per_class, axis=1)     # [N]

    return ova_mi_per_class, ova_mi_sum


# =============================================================================
# STRUCTURE-AWARE PROPOSED: C_k + Covariance
# =============================================================================

def compute_pairwise_correlation(mc_predictions, mu=None, var=None, eps=1e-10):
    """
    Compute pairwise Pearson correlation between class probabilities
    across stochastic forward passes.

        rho_{ij}(x) = Cov[p_i, p_j] / (std[p_i] * std[p_j])

    A strongly negative rho_{ij} means classes i and j trade probability
    mass across passes — the model alternates between "this is class i"
    and "this is class j", indicating epistemic confusion between i and j.

    Args:
        mc_predictions: np.ndarray [S, N, K] -- MC samples.
        mu:  np.ndarray [N, K] or None -- precomputed mean.
        var: np.ndarray [N, K] or None -- precomputed variance.
        eps: Small constant for numerical stability.

    Returns:
        corr: np.ndarray [N, K, K] -- per-sample correlation matrix.
        cov:  np.ndarray [N, K, K] -- per-sample covariance matrix.
    """
    S, N, K = mc_predictions.shape

    if mu is None:
        mu = np.mean(mc_predictions, axis=0)  # [N, K]
    if var is None:
        var = np.var(mc_predictions, axis=0)  # [N, K], ddof=0

    # Centered deviations: [S, N, K]
    delta = mc_predictions - mu[np.newaxis, :, :]

    # Covariance: [N, K, K] (use ddof=0 to match np.var default)
    cov = np.einsum('snk,snl->nkl', delta, delta) / S

    # Standard deviations: [N, K]
    std = np.sqrt(var + eps)

    # Correlation: [N, K, K]
    corr = cov / (std[:, :, np.newaxis] * std[:, np.newaxis, :] + eps)

    return corr, cov


def compute_cbec(C_k, corr, safe_classes, critical_classes, weights=None, eps=1e-10):
    """
    Cross-Boundary Epistemic Confusion (CBEC).

        CBEC(x) = sum_{i in S} sum_{j in C} sqrt(C_i * C_j) * max(0, -rho_{ij})

    Detects epistemic confusion specifically across the safe/critical
    boundary by combining:
      - sqrt(C_i * C_j): geometric mean of per-class epistemic uncertainty
        on both sides of the boundary. High only when BOTH classes carry
        substantial epistemic uncertainty. This is where C_k does real work:
        it filters out minority-class baseline inflation because C_j being
        high alone (from the 1/mu_k correction) is not enough — C_i on the
        other side must also be elevated.
      - max(0, -rho_{ij}): strength of negative correlation between classes
        i and j. Negative correlation means the model alternates between
        assigning probability to i vs j across stochastic passes — direct
        evidence of pairwise confusion. Clamped at 0 so positive correlations
        (classes that co-occur, not compete) contribute nothing.

    The score is mathematically equivalent to summing the off-diagonal
    cross-boundary block of the Fisher-normalized covariance matrix
    (negated), where the Fisher normalization 1/sqrt(mu_i * mu_j)
    extends the same boundary-suppression correction as C_k's 1/mu_k
    to pairwise covariance.

    Why each component is necessary:
      - Without sqrt(C_i * C_j): raw correlation doesn't capture epistemic
        *magnitude*. Two classes can be perfectly anticorrelated but with
        tiny variance — no real confusion.
      - Without (-rho_{ij}): C_k alone cannot distinguish independent
        fluctuations from genuine pairwise confusion. Two classes can both
        have high C_k for completely unrelated reasons.

    Scenario analysis:
      - Minority inflation (C_3 high baseline): rho_{03} ~ 0, score ~ 0. Correct.
      - Safe confusion (Grade 0 <-> 1): pair not in S x C, excluded. Correct.
      - Boundary confusion (Grade 0 <-> 3): C_0 moderate, C_3 high,
        rho_{03} << 0, score high. Correct: defer.
      - Independent epistemic (C_0 from 0<->1, C_3 from 2<->3):
        rho_{03} ~ 0, score ~ 0. Correct: no boundary confusion.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        corr: np.ndarray [N, K, K] -- pairwise Pearson correlation.
        safe_classes: list of int -- safe class indices (e.g. [0, 1]).
        critical_classes: list of int -- critical class indices (e.g. [2, 3]).
        weights: np.ndarray or None -- shape [len(safe), len(critical)],
            optional clinical risk weights for each (i,j) pair.
            If None, all cross-boundary pairs weighted equally.
            Example for DR: w[0,1]=1.0 (Grade 0<->3), w[1,0]=0.5 (Grade 1<->2).
        eps: Small constant for numerical stability.

    Returns:
        cbec: np.ndarray [N] -- CBEC deferral score per sample.
            Higher = more cross-boundary epistemic confusion = defer first.
    """
    N = C_k.shape[0]
    cbec = np.zeros(N)

    for idx_i, i in enumerate(safe_classes):
        for idx_j, j in enumerate(critical_classes):
            # Geometric mean of per-class epistemic uncertainty
            geom_mean = np.sqrt(C_k[:, i] * C_k[:, j] + eps)

            # Strength of negative correlation (clamped at 0)
            neg_corr = np.maximum(0.0, -corr[:, i, j])

            # Optional clinical risk weight
            w = weights[idx_i, idx_j] if weights is not None else 1.0

            cbec += w * geom_mean * neg_corr

    return cbec




# =============================================================================
# All-in-One Computation
# =============================================================================

def compute_all_uncertainties(mc_predictions, critical_classes=None, safe_classes=None):
    """
    Compute ALL uncertainty metrics from MC predictions.

    Includes proposed C_k methods, scalar baselines, and per-class baselines.

    Args:
        mc_predictions: np.ndarray [S, N, K] -- S stochastic forward passes.
        critical_classes: list of int -- critical class indices (default [3]).
        safe_classes: list of int -- safe class indices (default [0, 1]).

    Returns:
        dict with keys:

        Predictive statistics:
            'mu'                [N, K]  predictive mean
            'var'               [N, K]  predictive variance
            'm3'                [N, K]  third central moment

        Proposed (C_k-based):
            'C_k'               [N, K]  per-class epistemic uncertainty
            'rho_k'             [N, K]  skewness diagnostic
            'C_critical_sum'    [N]     sum of C_k for critical classes
            'C_critical_max'    [N]     max of C_k for critical classes
            'sum_C_k'           [N]     sum of all C_k (theory: ~ MI)

        Scalar baselines:
            'entropy'           [N]     predictive entropy H[mu]
            'MI'                [N]     mutual information
            'maxprob'           [N]     1 - max(mu_k)

        Per-class baselines:
            'var_critical_max'  [N]     max Var[p_k] over critical classes
            'sale_eu'           [N, K]  Sale et al. label-wise EU (= Var[p_k])
            'sale_eu_global'    [N]     Sale global EU = sum_k Var[p_k]
            'sale_eu_critical'  [N]     Sale EU on critical classes
            'ova_mi_per_class'  [N, len(crit)]  one-vs-all MI per critical class
            'ova_mi_sum'        [N]     sum of one-vs-all MI (deferral score)

        Structure-aware proposed (C_k + covariance):
            'corr'              [N, K, K]  pairwise Pearson correlation
            'cov'               [N, K, K]  pairwise covariance
            'cbec'              [N]        Cross-Boundary Epistemic Confusion

        Predictions:
            'y_pred'            [N]     predicted class from mean
    """
    if critical_classes is None:
        critical_classes = [3]
    if safe_classes is None:
        safe_classes = [0, 1]

    # --- Predictive statistics ---
    mu, var, m3 = compute_predictive_statistics(mc_predictions)

    # --- Proposed: C_k and diagnostics ---
    C_k = compute_per_class_epistemic(mu, var)
    rho_k = compute_skewness_diagnostic(mu, var, m3)
    C_crit_sum = compute_critical_epistemic_sum(C_k, critical_classes)
    C_crit_max = compute_critical_epistemic_max(C_k, critical_classes)
    sum_C_k = np.sum(C_k, axis=1)

    # --- Scalar baselines ---
    entropy = compute_predictive_entropy(mu)
    mi = compute_mutual_information(mc_predictions, mu=mu)
    maxprob = compute_maxprob_uncertainty(mu)

    # --- Per-class baselines ---
    var_crit_max = compute_critical_variance_max(var, critical_classes)
    sale_eu = compute_sale_eu(var)
    sale_eu_global = compute_sale_eu_global(var)
    sale_eu_critical = compute_sale_eu_critical(var, critical_classes)
    ova_mi_per_class, ova_mi_sum = compute_one_vs_all_mi(
        mc_predictions, critical_classes
    )

    # --- Structure-aware proposed (C_k + covariance) ---
    corr, cov = compute_pairwise_correlation(mc_predictions, mu=mu, var=var)
    cbec = compute_cbec(C_k, corr, safe_classes, critical_classes)

    y_pred = np.argmax(mu, axis=1)

    return {
        # Predictive statistics
        'mu': mu,
        'var': var,
        'm3': m3,
        # Proposed (C_k-based)
        'C_k': C_k,
        'rho_k': rho_k,
        'C_critical_sum': C_crit_sum,
        'C_critical_max': C_crit_max,
        'sum_C_k': sum_C_k,
        # Scalar baselines
        'entropy': entropy,
        'MI': mi,
        'maxprob': maxprob,
        # Per-class baselines
        'var_critical_max': var_crit_max,
        'sale_eu': sale_eu,
        'sale_eu_global': sale_eu_global,
        'sale_eu_critical': sale_eu_critical,
        'ova_mi_per_class': ova_mi_per_class,
        'ova_mi_sum': ova_mi_sum,
        # Structure-aware proposed (C_k + covariance)
        'corr': corr,
        'cov': cov,
        'cbec': cbec,
        # Predictions
        'y_pred': y_pred,
    }


# =============================================================================
# Diagnostic Analysis: Per-Class Epistemic Profiles and Monitoring
# =============================================================================

def compute_epistemic_profiles(C_k, y_true, normalize=True):
    """
    Per-class epistemic profiles: E[C_k(x) | y = i] for all (i, k).

    For each true class i, computes the average C_k vector across all
    samples with y_true = i. This reveals which classes drive epistemic
    uncertainty for each ground-truth category.

    Interpretation examples (DR):
      - E[C_3 | y=3] high, E[C_0 | y=3] high → model confuses Grade 3
        with Grade 0 (dangerous: crosses safe/critical boundary)
      - E[C_3 | y=3] high, E[C_2 | y=3] high → model confuses Grade 3
        with Grade 2 (less dangerous: both critical)
      - E[C_0 | y=0] low for all k → model is epistemically confident
        on Grade 0 (expected: majority class)

    This is strictly more informative than scalar MI because MI would
    just say "Grade 3 samples have high uncertainty" without revealing
    which alternative hypotheses the model entertains.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        y_true: np.ndarray [N] -- ground truth labels (int).
        normalize: bool -- if True, also return C_k as fraction of
            sum_k C_k for each sample (epistemic mass distribution).

    Returns:
        profiles: dict with keys:
            'mean': np.ndarray [num_classes, K] -- E[C_k | y=i]
            'std': np.ndarray [num_classes, K] -- Std[C_k | y=i]
            'median': np.ndarray [num_classes, K] -- Median[C_k | y=i]
            'count': np.ndarray [num_classes] -- number of samples per class
            'mean_normalized': np.ndarray [num_classes, K] or None --
                E[C_k / sum_j C_j | y=i], i.e. what fraction of total
                epistemic uncertainty is attributed to each class.
                Only returned if normalize=True.
    """
    K = C_k.shape[1]
    num_classes = K
    eps = 1e-10

    # Initialize output arrays
    mean = np.zeros((num_classes, K))
    std = np.zeros((num_classes, K))
    median = np.zeros((num_classes, K))
    count = np.zeros(num_classes, dtype=int)

    # Compute per-class statistics
    for i in range(num_classes):
        mask = (y_true == i)
        count[i] = mask.sum()
        if count[i] > 0:
            mean[i] = np.mean(C_k[mask], axis=0)
            std[i] = np.std(C_k[mask], axis=0)
            median[i] = np.median(C_k[mask], axis=0)

    # Compute normalized profiles
    mean_normalized = None
    if normalize:
        sum_Ck = np.sum(C_k, axis=1, keepdims=True) + eps
        C_k_frac = C_k / sum_Ck
        mean_normalized = np.zeros((num_classes, K))
        for i in range(num_classes):
            mask = (y_true == i)
            if count[i] > 0:
                mean_normalized[i] = np.mean(C_k_frac[mask], axis=0)

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'count': count,
        'mean_normalized': mean_normalized,
    }


def compute_error_epistemic_signatures(C_k, y_true, y_pred, mi=None):
    """
    Error-conditioned epistemic signatures: E[C_k | y=i, yhat=j] for all (i, j, k).

    For each confusion pair (true class i, predicted class j), computes
    the average C_k vector. Different error types have distinct C_k
    fingerprints that scalar MI collapses into a single number.

    C_k can distinguish:
      - "True 3, predicted 0" (catastrophic miss): expect high C_3 AND C_0,
        indicating the model entertained both hypotheses but chose wrong.
      - "True 3, predicted 2" (severity underestimate): expect high C_3 AND C_2,
        different fingerprint from the catastrophic case.
      - "True 0, predicted 0" (correct, confident): expect low C_k across board.
      - "True 0, predicted 0" (correct, uncertain): moderate C_k — the model
        got it right but was epistemically unsure.

    Scalar MI for two different Grade 3 misclassifications could be identical,
    but C_k signatures reveal completely different confusion patterns.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        y_true: np.ndarray [N] -- ground truth labels.
        y_pred: np.ndarray [N] -- predicted labels.
        mi: np.ndarray [N] or None -- scalar MI per sample (optional,
            for comparison: "same MI, different C_k signature").

    Returns:
        signatures: dict with keys:
            'mean': dict mapping (i, j) -> np.ndarray [K]
                Mean C_k vector for samples with y_true=i, y_pred=j.
            'std': dict mapping (i, j) -> np.ndarray [K]
                Std of C_k for each (i, j) pair.
            'count': dict mapping (i, j) -> int
                Number of samples in each (i, j) cell.
            'mean_mi': dict mapping (i, j) -> float or None
                Mean scalar MI for each (i, j) cell (if mi provided).
            'pairs': list of (i, j) tuples with count > 0.
    """
    K = C_k.shape[1]
    classes = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))

    mean_dict = {}
    std_dict = {}
    count_dict = {}
    mi_dict = {}
    pairs = []

    for i in classes:
        for j in classes:
            mask = (y_true == i) & (y_pred == j)
            n = mask.sum()
            if n > 0:
                count_dict[(i, j)] = int(n)
                mean_dict[(i, j)] = np.mean(C_k[mask], axis=0)
                std_dict[(i, j)] = np.std(C_k[mask], axis=0)
                if mi is not None:
                    mi_dict[(i, j)] = float(np.mean(mi[mask]))
                pairs.append((i, j))

    return {
        'mean': mean_dict,
        'std': std_dict,
        'count': count_dict,
        'mean_mi': mi_dict if mi is not None else None,
        'pairs': pairs,
    }


def compute_epistemic_dominance(C_k, eps=1e-10):
    """
    Epistemic dominance: which class dominates each sample's uncertainty.

    For each sample, computes:
      - The class with the largest C_k (dominant epistemic class)
      - The fraction of total MI attributed to that class
      - The entropy of the C_k distribution (epistemic concentration)

    A sample where one class dominates (high fraction, low entropy) has
    *focused* epistemic uncertainty — the model knows what it doesn't know.
    A sample with dispersed C_k (low fraction, high entropy) has
    *diffuse* epistemic uncertainty — the model is confused about everything.

    For selective prediction: focused uncertainty on a critical class
    (dominant class in {2,3}) is more informative for deferral decisions than diffuse uncertainty.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        eps: Small constant for numerical stability.

    Returns:
        dominance: dict with keys:
            'dominant_class': np.ndarray [N] -- argmax_k C_k for each sample.
            'dominant_fraction': np.ndarray [N] -- max_k C_k / sum_k C_k.
            'C_k_fractions': np.ndarray [N, K] -- C_k / sum_k C_k per sample.
            'epistemic_entropy': np.ndarray [N] -- H(C_k / sum_k C_k),
                entropy of the epistemic mass distribution. Low = concentrated,
                high = diffuse.
    """
    sum_Ck = np.sum(C_k, axis=1, keepdims=True) + eps
    fractions = C_k / sum_Ck
    dominant_class = np.argmax(C_k, axis=1)
    dominant_fraction = np.max(fractions, axis=1)
    epistemic_entropy = -np.sum(fractions * np.log(fractions + eps), axis=1)

    return {
        'dominant_class': dominant_class,
        'dominant_fraction': dominant_fraction,
        'C_k_fractions': fractions,
        'epistemic_entropy': epistemic_entropy,
    }


def compute_epistemic_drift(C_k_batches, y_pred_batches=None, class_names=None):
    """
    Per-class epistemic drift monitoring across deployment batches.

    Tracks E[C_k] and E[C_k | yhat=k] over time (batches). A spike in
    C_k for a specific class signals that the model is seeing inputs
    where class k's probability fluctuates across stochastic passes —
    i.e., the model is encountering class-k-like inputs it hasn't seen
    in training.

    Key advantage over scalar MI monitoring: MI spike just says "more
    uncertain." C_k drift tells you "uncertainty about Grade 3 spiked"
    or "uncertainty about Grade 0 spiked" — directly relevant for a
    hospital QA team deciding whether to retrain, recalibrate, or halt.

    No ground truth needed — this runs on live predictions.

    Args:
        C_k_batches: list of np.ndarray, each [N_b, K] -- C_k values
            for each deployment batch (time window).
        y_pred_batches: list of np.ndarray, each [N_b] or None --
            predicted labels per batch (optional, for conditional stats).
        class_names: list of str or None -- names for each class
            (e.g., ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']).

    Returns:
        drift: dict with keys:
            'batch_mean': np.ndarray [T, K] -- E[C_k] per batch.
            'batch_std': np.ndarray [T, K] -- Std[C_k] per batch.
            'batch_median': np.ndarray [T, K] -- Median[C_k] per batch.
            'batch_mi_approx': np.ndarray [T] -- E[sum_k C_k] per batch
                (approximate MI, for comparison with per-class trends).
            'conditional_mean': np.ndarray [T, K] or None --
                E[C_k | yhat=k] per batch (self-uncertainty: how uncertain
                is the model about class k when it predicts class k).
                Only computed if y_pred_batches is provided.
            'n_batches': int
            'class_names': list of str
    """
    T = len(C_k_batches)
    K = C_k_batches[0].shape[1]

    batch_mean = np.zeros((T, K))
    batch_std = np.zeros((T, K))
    batch_median = np.zeros((T, K))
    batch_mi_approx = np.zeros(T)

    for t in range(T):
        ck = C_k_batches[t]
        batch_mean[t] = np.mean(ck, axis=0)
        batch_std[t] = np.std(ck, axis=0)
        batch_median[t] = np.median(ck, axis=0)
        batch_mi_approx[t] = np.mean(np.sum(ck, axis=1))

    conditional_mean = None
    if y_pred_batches is not None:
        conditional_mean = np.full((T, K), np.nan)
        for t in range(T):
            for k in range(K):
                mask = (y_pred_batches[t] == k)
                if mask.sum() > 0:
                    conditional_mean[t, k] = np.mean(C_k_batches[t][mask, k])

    if class_names is None:
        class_names = [f'Class {k}' for k in range(K)]

    return {
        'batch_mean': batch_mean,
        'batch_std': batch_std,
        'batch_median': batch_median,
        'batch_mi_approx': batch_mi_approx,
        'conditional_mean': conditional_mean,
        'n_batches': T,
        'class_names': class_names,
    }


def compute_confusion_epistemic_matrix(C_k, corr, eps=1e-10):
    """
    Epistemic confusion matrix: average cross-class epistemic signal.

    For each class pair (i, j), computes:
        E[sqrt(C_i * C_j) * max(0, -rho_{ij})]

    This is the per-pair version of CBEC, averaged over all samples.
    The resulting K x K matrix shows which class pairs the model
    systematically confuses due to epistemic uncertainty.

    This matrix is a key diagnostic artifact for the paper:
      - Diagonal: zero by construction (rho_{ii} = 1, clamped to 0)
      - Off-diagonal: strength of epistemic confusion between i and j
      - Cross-boundary block (S x C): what CBEC aggregates for deferral
      - Within-group blocks (S x S, C x C): confusion that CBEC ignores

    Comparing the cross-boundary block magnitude to within-group blocks
    validates (or invalidates) the safe/critical grouping choice.

    Args:
        C_k: np.ndarray [N, K] -- per-class epistemic uncertainty.
        corr: np.ndarray [N, K, K] -- pairwise Pearson correlation.
        eps: Small constant for numerical stability.

    Returns:
        epi_confusion: np.ndarray [K, K] -- average epistemic confusion
            between each class pair. Symmetric, zero diagonal.
    """
    N, K = C_k.shape
    epi_confusion = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i != j:
                geom = np.sqrt(C_k[:, i] * C_k[:, j] + eps)
                neg_corr = np.maximum(0.0, -corr[:, i, j])
                epi_confusion[i, j] = np.mean(geom * neg_corr)

    return epi_confusion


# =============================================================================
# Monte Carlo Inference
# =============================================================================

def run_mc_inference(model, dataset, n_samples=100):
    """
    Run Monte Carlo stochastic forward passes through a Bayesian model.

    Each pass samples weights from their posterior (training=True).

    IMPORTANT: The @tf.function prediction must be compiled BEFORE the
    loop to avoid retracing on every call. We use tf.function with the
    model's __call__ via a concrete function to ensure single compilation.

    Args:
        model: Bayesian tf.keras.Model.
        dataset: tf.data.Dataset yielding (images, labels) batches.
        n_samples: Number of stochastic forward passes.

    Returns:
        mc_predictions: np.ndarray [n_samples, N, K].
        y_true: np.ndarray [N] -- ground truth labels (int).
    """
    import tensorflow as tf

    # Compile ONCE before the loop — this is critical for speed.
    # Defining @tf.function inside a function scope with `model` as
    # closure causes retracing. Instead, use model.call directly.
    mc_predict = tf.function(lambda x: model(x, training=True))

    # Warm up: trace the graph once with a real batch
    for x_batch, _ in dataset:
        _ = mc_predict(x_batch)
        break

    # Extract ground truth labels (only need to do this once)
    y_true = []
    for _, y_batch in dataset:
        y_true.append(np.argmax(y_batch.numpy(), axis=1))
    y_true = np.concatenate(y_true)

    # MC sampling
    mc_predictions = []
    for i in range(n_samples):
        print(f'  MC sample {i+1}/{n_samples}...', end='\r')

        batch_preds = []
        for x_batch, _ in dataset:
            pred = mc_predict(x_batch)
            batch_preds.append(pred.numpy())

        sample_preds = np.concatenate(batch_preds, axis=0)
        mc_predictions.append(sample_preds)

    N = len(y_true)
    mc_predictions = np.stack(mc_predictions, axis=0)[:, :N, :]

    print(f'\n  MC inference complete: {mc_predictions.shape}')
    return mc_predictions, y_true


# =============================================================================
# Selective Prediction Evaluation
# =============================================================================

def selective_prediction_eval(y_true, y_pred, uncertainty_scores,
                              coverage_levels, critical_classes=None):
    """
    Evaluate selective prediction at multiple coverage levels.

    Samples are ranked by uncertainty (low -> high). At coverage c,
    the model auto-decides on the top c fraction (most confident)
    and defers the rest to a human expert.

    Two critical-class safety metrics are computed:

        critical_fnr:  True false negative rate — fraction of critical
            samples predicted as SAFE (crosses the clinical boundary).
            A Grade 3 predicted as Grade 0 counts. A Grade 3 predicted
            as Grade 2 does NOT count (patient still gets treatment).
            This is the primary safety metric.

        critical_err:  Critical error rate — fraction of critical
            samples predicted as ANY wrong class (including within-
            critical errors like Grade 3 → Grade 2). This captures
            all misclassifications of critical patients regardless
            of whether the prediction stays in the critical group.

    Args:
        y_true: np.ndarray [N] -- ground truth labels.
        y_pred: np.ndarray [N] -- predicted labels.
        uncertainty_scores: np.ndarray [N] -- scalar uncertainty
            (higher = more uncertain = defer first).
        coverage_levels: list of float -- coverage fractions to evaluate.
        critical_classes: list of int -- class indices for FNR computation.

    Returns:
        list of dict, one per coverage level, with keys:
            'coverage':           float   coverage fraction
            'n_kept':             int     number of samples auto-decided
            'accuracy':           float   accuracy on kept samples
            'f1_macro':           float   macro F1 on kept samples
            'critical_fnr':       float   true FNR: critical → safe
            'critical_err':       float   critical error rate: any misclass.
            'critical_n':         int     number of critical samples kept
            'critical_defer_rate': float  fraction of all critical deferred
    """
    from sklearn.metrics import accuracy_score, f1_score

    if critical_classes is None:
        critical_classes = [3]

    N = len(y_true)
    sorted_idx = np.argsort(uncertainty_scores)
    all_critical_mask = np.isin(y_true, critical_classes)
    n_all_critical = all_critical_mask.sum()

    results = []
    for cov in coverage_levels:
        n_keep = int(np.ceil(cov * N))
        keep_idx = sorted_idx[:n_keep]

        yt = y_true[keep_idx]
        yp = y_pred[keep_idx]

        acc = accuracy_score(yt, yp)
        f1_m = f1_score(yt, yp, average='macro', zero_division=0)

        crit_mask = np.isin(yt, critical_classes)
        n_crit = crit_mask.sum()
        if n_crit > 0:
            # True FNR: critical sample predicted as safe (boundary crossing)
            pred_safe = ~np.isin(yp, critical_classes)
            crit_fnr = np.sum(pred_safe & crit_mask) / n_crit

            # Critical error rate: any misclassification of critical sample
            crit_err = np.sum((yt != yp) & crit_mask) / n_crit
        else:
            crit_fnr = 0.0
            crit_err = 0.0

        if n_all_critical > 0:
            deferred_idx = sorted_idx[n_keep:]
            crit_deferred = np.isin(y_true[deferred_idx],
                                    critical_classes).sum()
            crit_defer_rate = crit_deferred / n_all_critical
        else:
            crit_defer_rate = 0.0

        results.append({
            'coverage': cov,
            'n_kept': n_keep,
            'accuracy': acc,
            'f1_macro': f1_m,
            'critical_fnr': crit_fnr,
            'critical_err': crit_err,
            'critical_n': n_crit,
            'critical_defer_rate': crit_defer_rate,
        })

    return results


def compute_ausc(y_true, y_pred, uncertainty_scores, critical_classes=None,
                 n_points=200, metric='critical_fnr'):
    """
    Area Under Selective Risk Curve (AUSC).

    Lower AUSC = better selective prediction policy.

    Args:
        y_true: np.ndarray [N].
        y_pred: np.ndarray [N].
        uncertainty_scores: np.ndarray [N].
        critical_classes: list of int.
        n_points: Number of coverage points for integration.
        metric: str — which risk metric to integrate:
            'critical_fnr':  True false negative rate — critical samples
                predicted as safe (boundary crossing). Primary safety metric.
            'critical_err':  Critical error rate — any misclassification
                of critical samples (includes within-critical errors).
            'error_rate':    Overall misclassification rate.

    Returns:
        ausc: float -- area under the selective risk curve.
        coverages: np.ndarray -- coverage values used.
        risks: list -- risk at each coverage.
    """
    if critical_classes is None:
        critical_classes = [3]

    coverages = np.linspace(0.05, 1.0, n_points)
    sorted_idx = np.argsort(uncertainty_scores)
    N = len(y_true)

    risks = []
    for cov in coverages:
        n_keep = max(1, int(np.ceil(cov * N)))
        keep_idx = sorted_idx[:n_keep]
        yt = y_true[keep_idx]
        yp = y_pred[keep_idx]

        if metric == 'critical_fnr':
            # True FNR: critical → safe (boundary crossing)
            crit_mask = np.isin(yt, critical_classes)
            n_crit = crit_mask.sum()
            pred_safe = ~np.isin(yp, critical_classes)
            risk = (np.sum(pred_safe & crit_mask) / n_crit
                    if n_crit > 0 else 0.0)
        elif metric == 'critical_err':
            # Critical error rate: any misclassification of critical
            crit_mask = np.isin(yt, critical_classes)
            n_crit = crit_mask.sum()
            risk = (np.sum((yt != yp) & crit_mask) / n_crit
                    if n_crit > 0 else 0.0)
        elif metric == 'error_rate':
            risk = 1.0 - np.mean(yt == yp)
        else:
            raise ValueError(f'Unknown metric: {metric}. '
                             f'Choose from: critical_fnr, critical_err, error_rate')

        risks.append(risk)

    ausc = float(np.trapz(risks, coverages))
    return ausc, coverages, risks


# =============================================================================
# Build Deferral Policies
# =============================================================================

def build_deferral_policies(metrics_dict):
    """
    Build a dict of ALL named deferral policies from computed metrics.

    Each policy maps to a scalar uncertainty score per sample.
    Higher score = more uncertain = deferred first.

    Organized into four groups:
        - Scalar baselines (Entropy, MI, MaxProb)
        - Per-class baselines (Var_critical, Sale EU variants, OvA MI)
        - Proposed C_k methods (C_critical_sum, C_critical_max)
        - Structure-aware proposed (CBEC)

    Args:
        metrics_dict: dict returned by compute_all_uncertainties().

    Returns:
        dict {policy_name: np.ndarray [N]}.
    """
    return {
        # --- Scalar baselines ---
        'Entropy':          metrics_dict['entropy'],
        'MI':               metrics_dict['MI'],
        'MaxProb':          metrics_dict['maxprob'],
        # --- Per-class baselines ---
        'Var_critical':     metrics_dict['var_critical_max'],
        'Sale_EU_global':   metrics_dict['sale_eu_global'],
        'Sale_EU_critical': metrics_dict['sale_eu_critical'],
        'OvA_MI':           metrics_dict['ova_mi_sum'],
        # --- Proposed (C_k-based) ---
        'C_critical_sum':   metrics_dict['C_critical_sum'],
        'C_critical_max':   metrics_dict['C_critical_max'],
        # --- Structure-aware proposed (C_k + covariance) ---
        'CBEC':             metrics_dict['cbec'],
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ausc(y_true, y_pred, uncertainty_scores_dict, critical_classes,
                   n_bootstrap=200, seed=42, n_ausc_points=200,
                   metric='critical_fnr'):
    """
    Bootstrap confidence intervals for AUSC across multiple deferral policies.

    Resamples the test set WITH REPLACEMENT n_bootstrap times. For each
    bootstrap sample, recomputes AUSC for every policy. Reports mean,
    std, and 95% CI for each policy, plus pairwise p-values.

    This does NOT re-run model inference — it resamples from the already-
    computed uncertainty scores. This is the correct approach: we are
    measuring robustness of the metric ranking to test set composition,
    not to model randomness.

    Args:
        y_true: np.ndarray [N] — ground truth labels.
        y_pred: np.ndarray [N] — predicted labels.
        uncertainty_scores_dict: dict {policy_name: np.ndarray [N]}
            — all deferral policies with their uncertainty scores.
        critical_classes: list of int.
        n_bootstrap: int — number of bootstrap iterations (200 recommended).
        seed: int — random seed for reproducibility.
        n_ausc_points: int — coverage grid resolution for AUSC.
        metric: str — 'critical_fnr', 'critical_err', or 'error_rate'.

    Returns:
        bootstrap_results: dict with keys:
            'ausc_samples': dict {name: np.ndarray [n_bootstrap]}
                — AUSC value for each policy on each bootstrap sample.
            'summary': dict {name: {'mean': float, 'std': float,
                'ci_lower': float, 'ci_upper': float, 'median': float}}
            'pairwise_pvalues': dict {(name_a, name_b): float}
                — fraction of bootstrap samples where policy A has
                lower AUSC than policy B. If > 0.975, A is significantly
                better at 95% confidence.
            'ranking_stability': dict {name: float}
                — fraction of bootstrap samples where this policy
                has the lowest AUSC (i.e., wins).
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)

    # Initialize storage for bootstrap samples
    ausc_samples = {name: np.zeros(n_bootstrap) for name in uncertainty_scores_dict}

    print(f'Running bootstrap with {n_bootstrap} iterations...')
    for b in range(n_bootstrap):
        # Bootstrap resample indices
        idx = rng.choice(N, size=N, replace=True)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]

        # Compute AUSC for each policy on this bootstrap sample
        for name, scores in uncertainty_scores_dict.items():
            scores_b = scores[idx]
            ausc_b, _, _ = compute_ausc(yt_b, yp_b, scores_b, critical_classes,
                                       n_points=n_ausc_points, metric=metric)
            ausc_samples[name][b] = ausc_b

        # Print progress
        if (b + 1) % 50 == 0 or b == 0:
            print(f'  Bootstrap {b+1}/{n_bootstrap}', end='\r')

    print(f'\n  Bootstrap complete!')

    # Compute summary statistics for each policy
    summary = {}
    for name in uncertainty_scores_dict:
        samples = ausc_samples[name]
        summary[name] = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_lower': float(np.percentile(samples, 2.5)),
            'ci_upper': float(np.percentile(samples, 97.5)),
            'median': float(np.median(samples)),
        }

    # Compute pairwise p-values (fraction where A < B)
    pairwise_pvalues = {}
    policy_names = list(uncertainty_scores_dict.keys())
    for i, name_a in enumerate(policy_names):
        for name_b in policy_names[i+1:]:
            p = float(np.mean(ausc_samples[name_a] < ausc_samples[name_b]))
            pairwise_pvalues[(name_a, name_b)] = p

    # Compute ranking stability (fraction of times each policy wins)
    ranking_stability = {}
    all_ausc = np.stack([ausc_samples[name] for name in policy_names], axis=1)  # [n_bootstrap, n_policies]
    best_idx = np.argmin(all_ausc, axis=1)  # [n_bootstrap]
    for i, name in enumerate(policy_names):
        ranking_stability[name] = float(np.mean(best_idx == i))

    return {
        'ausc_samples': ausc_samples,
        'summary': summary,
        'pairwise_pvalues': pairwise_pvalues,
        'ranking_stability': ranking_stability,
    }


def bootstrap_selective_prediction(y_true, y_pred, uncertainty_scores_dict,
                                   coverage, critical_classes,
                                   n_bootstrap=200, seed=42):
    """
    Bootstrap CIs for selective prediction metrics at a specific coverage level.

    For each bootstrap resample, computes accuracy, macro F1, critical FNR
    (true: critical → safe) and critical error rate (any misclassification)
    at the given coverage for every policy.

    Args:
        y_true: np.ndarray [N].
        y_pred: np.ndarray [N].
        uncertainty_scores_dict: dict {policy_name: np.ndarray [N]}.
        coverage: float — single coverage level (e.g. 0.80).
        critical_classes: list of int.
        n_bootstrap: int.
        seed: int.

    Returns:
        dict {policy_name: {
            'accuracy': {'mean', 'std', 'ci_lower', 'ci_upper'},
            'critical_fnr': {'mean', 'std', 'ci_lower', 'ci_upper'},
            'critical_err': {'mean', 'std', 'ci_lower', 'ci_upper'},
            'f1_macro': {'mean', 'std', 'ci_lower', 'ci_upper'},
        }}
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)

    # Initialize storage
    policy_names = list(uncertainty_scores_dict.keys())
    results = {name: {
        'accuracy': [],
        'critical_fnr': [],
        'critical_err': [],
        'f1_macro': [],
    } for name in policy_names}

    print(f'Running bootstrap at coverage={coverage:.2f} with {n_bootstrap} iterations...')
    for b in range(n_bootstrap):
        # Bootstrap resample
        idx = rng.choice(N, size=N, replace=True)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]

        # Evaluate each policy at this coverage
        for name, scores in uncertainty_scores_dict.items():
            scores_b = scores[idx]
            eval_results = selective_prediction_eval(
                yt_b, yp_b, scores_b, [coverage],
                critical_classes=critical_classes
            )
            r = eval_results[0]

            results[name]['accuracy'].append(r['accuracy'])
            results[name]['f1_macro'].append(r['f1_macro'])
            # Handle case where no critical samples are kept
            if r['critical_n'] > 0:
                results[name]['critical_fnr'].append(r['critical_fnr'])
                results[name]['critical_err'].append(r['critical_err'])
            else:
                results[name]['critical_fnr'].append(np.nan)
                results[name]['critical_err'].append(np.nan)

        if (b + 1) % 50 == 0 or b == 0:
            print(f'  Bootstrap {b+1}/{n_bootstrap}', end='\r')

    print(f'\n  Bootstrap complete!')

    # Compute summary statistics
    summary = {}
    for name in policy_names:
        summary[name] = {}
        for metric_name in ['accuracy', 'critical_fnr', 'critical_err', 'f1_macro']:
            samples = np.array(results[name][metric_name])
            summary[name][metric_name] = {
                'mean': float(np.nanmean(samples)),
                'std': float(np.nanstd(samples)),
                'ci_lower': float(np.nanpercentile(samples, 2.5)),
                'ci_upper': float(np.nanpercentile(samples, 97.5)),
            }

    return summary
