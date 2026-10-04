"""Effect-size computation for the rTMS meta-analysis.

All effects are standardized mean differences (Hedges' g) oriented so that
**positive = active rTMS better than sham**. Binary outcomes are converted from a
log odds ratio with the logistic approximation (Chinn, 2000), so continuous and
binary trials can be pooled on one scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Effect:
    g: float
    var: float
    method: str

    @property
    def se(self) -> float:
        return math.sqrt(self.var)

    def ci(self, z: float = 1.959964) -> tuple[float, float]:
        return self.g - z * self.se, self.g + z * self.se


def hedges_j(df: float) -> float:
    """Small-sample correction factor J."""
    return 1 - 3 / (4 * df - 1)


def _var_d(d: float, n1: int, n2: int) -> float:
    return (n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2))


def _to_g(d: float, n1: int, n2: int, method: str) -> Effect:
    j = hedges_j(n1 + n2 - 2)
    return Effect(g=j * d, var=j * j * _var_d(d, n1, n2), method=method)


def from_means(m_active: float, sd_active: float, n_active: int, m_sham: float,
               sd_sham: float, n_sham: int, lower_is_better: bool = True) -> Effect:
    sp = math.sqrt(((n_active - 1) * sd_active**2 + (n_sham - 1) * sd_sham**2)
                   / (n_active + n_sham - 2))
    diff = (m_sham - m_active) if lower_is_better else (m_active - m_sham)
    return _to_g(diff / sp, n_active, n_sham, "means/SD")


def from_t(t: float, n_active: int, n_sham: int) -> Effect:
    return _to_g(t * math.sqrt(1 / n_active + 1 / n_sham), n_active, n_sham, "t statistic")


def from_f(f: float, n_active: int, n_sham: int, sign: float = 1.0) -> Effect:
    """F(1, df) for a two-group comparison; ``sign`` gives the direction."""
    e = from_t(math.copysign(math.sqrt(f), sign), n_active, n_sham)
    return Effect(e.g, e.var, "F(1,df)")


def from_d(d: float, n_active: int, n_sham: int, already_g: bool = False) -> Effect:
    if already_g:
        return Effect(g=d, var=_var_d(d, n_active, n_sham), method="reported g")
    return _to_g(d, n_active, n_sham, "reported d")


def from_partial_eta2(eta2: float, n_active: int, n_sham: int, sign: float = 1.0) -> Effect:
    d = 2 * math.sqrt(eta2 / (1 - eta2))
    return _to_g(math.copysign(d, sign), n_active, n_sham, "partial eta^2")


def from_events(e_active: int, n_active: int, e_sham: int, n_sham: int,
                higher_is_better: bool = True) -> Effect:
    """Binary outcome -> log OR (0.5 continuity correction if needed) -> SMD."""
    a, b, c, d = e_active, n_active - e_active, e_sham, n_sham - e_sham
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((a * d) / (b * c))
    var_lor = 1 / a + 1 / b + 1 / c + 1 / d
    if not higher_is_better:
        log_or = -log_or
    k = math.sqrt(3) / math.pi
    return Effect(g=log_or * k, var=var_lor * k * k, method="events (logOR->SMD)")


def smd_to_log_odds(g: float) -> float:
    """Inverse of the logistic approximation: SMD -> log odds ratio."""
    return g * math.pi / math.sqrt(3)


def compute_effect(rec: dict) -> Effect | None:
    """Compute an Effect from one extracted record (see data/meta/*.csv schema)."""
    n1, n2 = rec.get("n_active"), rec.get("n_sham")
    sign = -1.0 if rec.get("direction") == "favors_sham" else 1.0
    if rec.get("mean_active") is not None and rec.get("sd_active"):
        return from_means(rec["mean_active"], rec["sd_active"], n1, rec["mean_sham"],
                          rec["sd_sham"], n2, lower_is_better=rec.get("lower_is_better", True))
    if rec.get("events_active") is not None:
        return from_events(rec["events_active"], n1, rec["events_sham"], n2,
                           higher_is_better=rec.get("higher_is_better", True))
    etype, val = rec.get("effect_type"), rec.get("effect_value")
    if etype is None or val is None:
        return None
    val = float(val)
    if etype in ("d", "g"):
        return from_d(sign * abs(val), n1, n2, already_g=etype == "g")
    if etype == "t":
        return from_t(sign * abs(val), n1, n2)
    if etype == "F":
        return from_f(abs(val), n1, n2, sign=sign)
    if etype == "eta_p2":
        return from_partial_eta2(abs(val), n1, n2, sign=sign)
    raise ValueError(f"unknown effect type {etype}")
