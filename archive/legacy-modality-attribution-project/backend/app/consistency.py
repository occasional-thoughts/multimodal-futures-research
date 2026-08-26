from itertools import combinations

import numpy as np
from scipy import stats

from app.config import CONSISTENCY_THRESHOLD


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_resamples: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) < 2:
        stat = float(statistic(values)) if len(values) else float("nan")
        return stat, stat, stat
    boots = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(statistic(values)), float(lo), float(hi)


def fusion_consistency(news_share_by_variant: dict, regime_mask: np.ndarray) -> dict:
    """Instance-level Spearman correlation of news_share between each pair of fusion
    variants, matched on the same test instances within one regime. This replaces
    correlating two aggregate means (which has no statistical power) with a
    per-instance comparison across many matched points.
    """
    variants = list(news_share_by_variant.keys())
    result = {}
    for a, b in combinations(variants, 2):
        xa = news_share_by_variant[a][regime_mask]
        xb = news_share_by_variant[b][regime_mask]
        if len(xa) < 3:
            result[f"{a}_vs_{b}"] = {"rho": float("nan"), "p": float("nan")}
            continue
        rho, p = stats.spearmanr(xa, xb)
        result[f"{a}_vs_{b}"] = {"rho": float(rho), "p": float(p)}
    return result


def regime_sensitivity(news_share: np.ndarray, calm_mask: np.ndarray, volatile_mask: np.ndarray) -> dict:
    """Does this fusion variant's attribution actually shift between market regimes?
    A significant difference is evidence the explanation tracks real market behavior,
    not a failure of consistency -- consistency is about fusion architecture, not regime.
    """
    calm_vals, vol_vals = news_share[calm_mask], news_share[volatile_mask]
    if len(calm_vals) < 3 or len(vol_vals) < 3:
        return {"u_stat": float("nan"), "p": float("nan"), "calm_mean": float("nan"), "volatile_mean": float("nan")}
    u_stat, p = stats.mannwhitneyu(calm_vals, vol_vals, alternative="two-sided")
    return {
        "u_stat": float(u_stat),
        "p": float(p),
        "calm_mean": float(calm_vals.mean()),
        "volatile_mean": float(vol_vals.mean()),
    }


def build_consistency_report(news_share_by_variant: dict, regime_labels: np.ndarray) -> dict:
    """Full report: fusion-consistency per regime + regime-sensitivity per variant,
    bootstrap CIs, and the verdict against the pre-registered threshold (config.CONSISTENCY_THRESHOLD).
    """
    report = {"by_regime": {}, "regime_sensitivity": {}, "threshold": CONSISTENCY_THRESHOLD}
    all_rhos = []

    for regime in ["calm", "volatile"]:
        mask = regime_labels == regime
        cons = fusion_consistency(news_share_by_variant, mask)
        for pair, stat_dict in cons.items():
            if not np.isnan(stat_dict["rho"]):
                all_rhos.append(stat_dict["rho"])
        report["by_regime"][regime] = cons

    for variant, shares in news_share_by_variant.items():
        calm_mask = regime_labels == "calm"
        vol_mask = regime_labels == "volatile"
        report["regime_sensitivity"][variant] = regime_sensitivity(shares, calm_mask, vol_mask)

    if all_rhos:
        mean_rho, lo, hi = bootstrap_ci(np.array(all_rhos))
        report["overall_consistency"] = {"mean_rho": mean_rho, "ci_low": lo, "ci_high": hi}
        report["verdict"] = "architecture-robust" if min(all_rhos) >= CONSISTENCY_THRESHOLD else "architecture-artifact"
    else:
        report["overall_consistency"] = {"mean_rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        report["verdict"] = "insufficient-data"

    return report
