"""Random-effects meta-analysis: DerSimonian-Laird and REML, Hartung-Knapp CIs,
heterogeneity, prediction intervals, Egger's test, leave-one-out and meta-regression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class MetaResult:
    estimate: float
    se: float
    ci: tuple[float, float]
    pi: tuple[float, float]  # 95% prediction interval for a new study
    p_value: float
    tau2: float
    i2: float
    q: float
    q_p: float
    k: int
    method: str
    weights: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "k": self.k, "method": self.method, "g": round(self.estimate, 4),
            "se": round(self.se, 4), "ci95": [round(x, 4) for x in self.ci],
            "pi95": [round(x, 4) for x in self.pi], "p": float(f"{self.p_value:.3g}"),
            "tau2": round(self.tau2, 4), "I2": round(self.i2, 3),
            "Q": round(self.q, 3), "Q_p": float(f"{self.q_p:.3g}"),
        }


def _q_stat(y: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - mu) ** 2)), mu


def tau2_dl(y: np.ndarray, v: np.ndarray) -> float:
    q, _ = _q_stat(y, v)
    w = 1 / v
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return max(0.0, (q - (len(y) - 1)) / c)


def tau2_reml(y: np.ndarray, v: np.ndarray, tol: float = 1e-8, max_iter: int = 200) -> float:
    tau2 = tau2_dl(y, v)
    for _ in range(max_iter):
        w = 1 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        # fixed-point REML update (Viechtbauer, 2005)
        new = max(0.0, np.sum(w**2 * ((y - mu) ** 2 - v)) / np.sum(w**2) + 1 / np.sum(w))
        if abs(new - tau2) < tol:
            return float(new)
        tau2 = new
    return float(tau2)


def random_effects(y, v, method: str = "REML", hartung_knapp: bool = True) -> MetaResult:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 2:
        raise ValueError("need at least two studies")
    tau2 = tau2_reml(y, v) if method.upper() == "REML" else tau2_dl(y, v)
    w = 1 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    q, _ = _q_stat(y, v)
    if hartung_knapp:
        # Hartung-Knapp with the "ad hoc" floor: never narrower than the standard SE
        se = float(max(np.sqrt(np.sum(w * (y - mu) ** 2) / ((k - 1) * np.sum(w))),
                       np.sqrt(1 / np.sum(w))))
        crit = stats.t.ppf(0.975, k - 1)
        p = 2 * stats.t.sf(abs(mu / se), k - 1)
    else:
        se = float(np.sqrt(1 / np.sum(w)))
        crit = stats.norm.ppf(0.975)
        p = 2 * stats.norm.sf(abs(mu / se))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    t_pi = stats.t.ppf(0.975, max(1, k - 2))
    pi_half = t_pi * np.sqrt(tau2 + se**2)
    return MetaResult(
        estimate=mu, se=se, ci=(mu - crit * se, mu + crit * se),
        pi=(mu - pi_half, mu + pi_half), p_value=float(p), tau2=tau2, i2=i2, q=q,
        q_p=float(stats.chi2.sf(q, k - 1)), k=k,
        method=f"{method.upper()}{' + Hartung-Knapp' if hartung_knapp else ''}",
        weights=w / w.sum(),
    )


def egger_test(y, v) -> dict:
    """Egger's regression test for small-study effects (funnel asymmetry)."""
    y, se = np.asarray(y, float), np.sqrt(np.asarray(v, float))
    res = stats.linregress(1 / se, y / se)
    t = res.intercept / res.intercept_stderr
    p = 2 * stats.t.sf(abs(t), len(y) - 2)
    return {"intercept": float(res.intercept), "p": float(p), "k": len(y)}


def leave_one_out(y, v, labels: list[str], **kw) -> list[dict]:
    y, v = np.asarray(y, float), np.asarray(v, float)
    out = []
    for i, lab in enumerate(labels):
        m = np.ones(len(y), bool)
        m[i] = False
        r = random_effects(y[m], v[m], **kw)
        out.append({"omitted": lab, "g": r.estimate, "ci95": r.ci, "I2": r.i2})
    return out


def meta_regression(y, v, x, names: list[str]) -> dict:
    """Mixed-effects meta-regression (method-of-moments tau^2, WLS coefficients).

    ``x`` is a (k, p) moderator matrix without intercept. Returns ``None`` when the
    design is rank deficient (collinear moderators) or leaves too few residual df.
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    k, p = X.shape
    if np.linalg.matrix_rank(X) < p or k - p < 2:
        return None
    W0 = np.diag(1 / v)
    beta0 = np.linalg.solve(X.T @ W0 @ X, X.T @ W0 @ y)
    resid = y - X @ beta0
    q_e = float(resid @ W0 @ resid)
    P = W0 - W0 @ X @ np.linalg.solve(X.T @ W0 @ X, X.T @ W0)
    tau2 = max(0.0, (q_e - (k - p)) / np.trace(P))
    W = np.diag(1 / (v + tau2))
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * stats.norm.sf(np.abs(z))
    return {
        "tau2_residual": float(tau2),
        "coefficients": {n: {"beta": float(b), "se": float(s), "p": float(pv)}
                         for n, b, s, pv in zip(["intercept", *names], beta, se, pvals)},
        "k": k,
    }
