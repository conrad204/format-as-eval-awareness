"""Purpose-axis fitting primitives and the implicit-purpose estimands.

The axis is the frozen recipe from `docs/analysis_plan_v2.md`: StandardScaler
plus LogisticRegression(C=0.1, max_iter=3000, random_state=0) on the
last-prompt-token residual stream, trained on the *decorrelated* cued stratum
(all four purpose x format cells) with leave-one-cue-vocabulary-family-out
splits. No-cue items carry no fold, so - following the same plan - they are
scored by the mean of the five held-fold models.

Everything downstream is expressed at the level of payload blocks, which are
the independent unit; uncertainty is a cluster bootstrap over the 320 blocks.
"""

from __future__ import annotations

import functools

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

PROBE_KW = dict(C=0.1, max_iter=3000, random_state=0)

# Estimator-robustness comparators. The point of this repository's own
# estimator-reversal result is that a substantive conclusion on this freeze can
# flip between probe pipelines, so the headline effect has to be re-measured
# under estimators with genuinely different inductive biases.
#
# `slda` and `svc` are chosen because they match the frozen logistic recipe in
# *quality* on this task (held-fold purpose AUC 0.983 and 0.988 against 0.989 at
# 8B L11) while differing in kind: a generative full-covariance discriminant and
# a max-margin hinge loss. `dlda` is the estimator from the repository's own
# reversal analysis, but it only reaches AUC 0.74 here because it ignores
# feature correlations, so a divergence under it is expected and weak evidence.
ALT_PROBES = {
    "slda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "svc": lambda: LinearSVC(C=0.01, max_iter=20000),
}
PROBE_CHOICES = ("logistic", "slda", "svc", "dlda")


def pinned(fn):
    """Run a worker with BLAS pinned to one thread.

    Without this the number of BLAS threads depends on `--jobs`, which changes
    the summation order in lbfgs and makes a layer recomputed on resume differ
    from the original in the 4th decimal of AUC. Pinning makes a resumed layer
    reproduce the value it would have had in the original run, and also stops
    joblib workers from oversubscribing the machine.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with threadpool_limits(limits=1):
            return fn(*args, **kwargs)
    return wrapper


DLDA_RIDGE = 0.1


def _fit(Z, y):
    clf = LogisticRegression(**PROBE_KW).fit(Z, y)
    if not np.all(clf.n_iter_ < PROBE_KW["max_iter"]):
        raise RuntimeError("probe did not converge")
    return clf


def diag_lda_direction(X, y, ridge=DLDA_RIDGE):
    """Diagonal LDA, reimplemented to match `scripts/full320_strict_equal_rows.py`.

    This repository has already shown that this freeze can produce *sign
    reversals* between L2 logistic and diagonal LDA
    (`full320_exact_logistic_estimator_reversal_conclusion_20260823`), so the
    comparison is only meaningful against that exact estimator: the same
    internal scaling by sqrt(pooled + delta^2/4), the same ridge as a fraction
    of the median positive scaled variance, the same primary ridge of 0.1, and
    no row normalisation.

    It also inherits that implementation's exact-class-balance requirement, and
    it operates on raw activations rather than StandardScaler output, because
    it does its own scaling.

    Returns (weight, intercept) in the raw feature space.
    """
    c0 = np.asarray(X[y == 0], dtype=np.float32)
    c1 = np.asarray(X[y == 1], dtype=np.float32)
    if len(c0) != len(c1):
        raise RuntimeError("diagonal LDA requires exact class balance")
    mu0 = c0.mean(axis=0, dtype=np.float64)
    mu1 = c1.mean(axis=0, dtype=np.float64)
    pooled = 0.5 * (np.var(c0, axis=0, dtype=np.float64)
                    + np.var(c1, axis=0, dtype=np.float64))
    delta = mu1 - mu0
    global_mean = 0.5 * (mu0 + mu1)
    global_var = pooled + 0.25 * delta * delta
    scale = np.sqrt(np.maximum(global_var, np.finfo(np.float64).tiny))
    pooled_scaled = pooled / (scale * scale)
    positive = pooled_scaled[np.isfinite(pooled_scaled) & (pooled_scaled > 0)]
    if not len(positive):
        raise RuntimeError("no positive pooled variances")
    variance = pooled_scaled + float(ridge) * float(np.median(positive))

    delta_scaled = delta / scale
    midpoint_scaled = (0.5 * (mu1 + mu0) - global_mean) / scale
    weight_scaled = delta_scaled / variance
    intercept = -float(np.dot(weight_scaled, midpoint_scaled))
    weight_raw = weight_scaled / scale
    raw_intercept = intercept - float(np.dot(weight_raw, global_mean))
    return weight_raw, raw_intercept


def _unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise RuntimeError("zero-norm direction")
    return v / n


def inlp_erase(Ztr, fmt_tr, Zs, max_rounds=12, target_auc=0.55, Zte=None, fmt_te=None):
    """Iterative nullspace projection of the linear *format* direction.

    Removes format directions from the standardized space until a freshly
    trained format probe is at (or below) `target_auc` on held-fold rows.
    Returns the erased training matrix, the erased versions of every matrix in
    `Zs`, and a diagnostic record. The purpose probe is refit afterwards on the
    erased space, so any surviving no-cue drift cannot be carried by the linear
    format subspace that was removed.
    """
    Ztr = Ztr.copy()
    Zs = [Z.copy() for Z in Zs]
    Zte = None if Zte is None else Zte.copy()
    aucs = []
    rounds = 0
    for _ in range(max_rounds):
        clf = _fit(Ztr, fmt_tr)
        if Zte is not None:
            aucs.append(float(roc_auc_score(fmt_te, clf.decision_function(Zte))))
            if aucs[-1] <= target_auc:
                break
        d = _unit(clf.coef_.ravel()).astype(Ztr.dtype)
        Ztr -= np.outer(Ztr @ d, d)
        Zs = [Z - np.outer(Z @ d, d) for Z in Zs]
        if Zte is not None:
            Zte -= np.outer(Zte @ d, d)
        rounds += 1
    else:
        if Zte is not None:
            clf = _fit(Ztr, fmt_tr)
            aucs.append(float(roc_auc_score(fmt_te, clf.decision_function(Zte))))
    return Ztr, Zs, Zte, {"rounds": rounds, "format_auc_trace": aucs,
                          "final_format_auc": aucs[-1] if aucs else None,
                          "reached_target": bool(aucs and aucs[-1] <= target_auc)}


def held_fold_scores(Xc, Xn, d, *, erase_format=False, permute_seed=None,
                     train_fmt=None, erase_kw=None, probe="logistic"):
    """Out-of-fold cued scores and no-cue scores at one layer.

    train_fmt: None trains on both format families (the decorrelated recipe);
    1 or 0 restricts training rows to benchmark / casual templates, which gives
    the cross-format transfer readouts.

    probe: "logistic" is the frozen recipe. "dlda" swaps in diagonal LDA, which
    skips the StandardScaler because it standardizes internally; it exists to
    test whether the conclusions are estimator-dependent, which this freeze is
    known to be capable of.
    """
    if probe == "dlda" and (erase_format or train_fmt is not None):
        raise ValueError("dlda path supports only the plain decorrelated recipe")
    oof = np.full(len(Xc), np.nan, dtype=np.float64)
    none_by_fold = np.zeros((d.n_folds, len(Xn)), dtype=np.float64)
    diag = {"purpose_auc_by_fold": [], "format_auc_by_fold": [], "erasure": []}
    rng = None if permute_seed is None else np.random.default_rng(permute_seed)

    for f in range(d.n_folds):
        tr = d.cued_fold != f
        te = ~tr
        if train_fmt is not None:
            tr = tr & (d.cued_fmt == train_fmt)

        y = d.cued_y.copy()
        if rng is not None:
            # Permute labels *within* each format cell of the training rows, so
            # the control destroys purpose while preserving every nuisance
            # marginal the real probe could exploit.
            for fv in (0, 1):
                cell = np.where(tr & (d.cued_fmt == fv))[0]
                y[cell] = y[cell][rng.permutation(len(cell))]

        if probe == "dlda":
            # Operates on raw activations: it standardizes internally.
            w, b = diag_lda_direction(Xc[tr], y[tr])
            oof[te] = Xc[te] @ w + b
            none_by_fold[f] = Xn @ w + b
            diag["purpose_auc_by_fold"].append(
                float(roc_auc_score(d.cued_y[te], oof[te])))
            continue

        sc = StandardScaler().fit(Xc[tr])
        Ztr = sc.transform(Xc[tr])
        Zte = sc.transform(Xc[te])
        Zn = sc.transform(Xn)

        if probe in ALT_PROBES:
            est = ALT_PROBES[probe]().fit(Ztr, y[tr])
            oof[te] = est.decision_function(Zte)
            none_by_fold[f] = est.decision_function(Zn)
            diag["purpose_auc_by_fold"].append(
                float(roc_auc_score(d.cued_y[te], oof[te])))
            continue

        fmt_clf = _fit(Ztr, d.cued_fmt[tr]) if train_fmt is None else None
        if fmt_clf is not None:
            diag["format_auc_by_fold"].append(
                float(roc_auc_score(d.cued_fmt[te], fmt_clf.decision_function(Zte)))
            )

        if erase_format:
            if train_fmt is not None:
                raise ValueError("format erasure requires decorrelated training")
            Ztr, (Zn,), Zte, erec = inlp_erase(
                Ztr, d.cued_fmt[tr], [Zn], Zte=Zte, fmt_te=d.cued_fmt[te],
                **(erase_kw or {})
            )
            diag["erasure"].append(erec)

        clf = _fit(Ztr, y[tr])
        oof[te] = clf.decision_function(Zte)
        none_by_fold[f] = clf.decision_function(Zn)
        diag["purpose_auc_by_fold"].append(
            float(roc_auc_score(d.cued_y[te], oof[te]))
        )

    return oof, none_by_fold.mean(axis=0), diag


def abstraction_gap(Xc, d, *, half_seed=0):
    """Within-vocabulary vs cross-vocabulary AUC at matched training set.

    Blocks are split into two halves. For each held fold f the probe trains on
    (folds != f, blocks in half A) and is scored twice on *the same* held-out
    blocks (half B): once on rows whose cue vocabulary was seen in training and
    once on rows from the held fold. The two readouts differ only in whether
    the test cue lexicon was observed, so their gap isolates how much of the
    decoding is carried by cue wording rather than an abstracted contrast.
    """
    rng = np.random.default_rng(half_seed)
    perm = rng.permutation(d.n_blocks)
    half_a = np.zeros(d.n_blocks, dtype=bool)
    half_a[perm[: d.n_blocks // 2]] = True
    a_rows = half_a[d.cued_block]

    within, cross = [], []
    for f in range(d.n_folds):
        tr = (d.cued_fold != f) & a_rows
        te_within = (d.cued_fold != f) & ~a_rows
        te_cross = (d.cued_fold == f) & ~a_rows
        sc = StandardScaler().fit(Xc[tr])
        clf = _fit(sc.transform(Xc[tr]), d.cued_y[tr])
        within.append(float(roc_auc_score(d.cued_y[te_within],
                                          clf.decision_function(sc.transform(Xc[te_within])))))
        cross.append(float(roc_auc_score(d.cued_y[te_cross],
                                         clf.decision_function(sc.transform(Xc[te_cross])))))
    return {"within_vocab_auc": float(np.mean(within)),
            "cross_vocab_auc": float(np.mean(cross)),
            "gap": float(np.mean(within) - np.mean(cross)),
            "within_by_fold": within, "cross_by_fold": cross}


# --------------------------------------------------------------------------
# Block-level estimands
# --------------------------------------------------------------------------

def _block_means(values, blocks, mask, n_blocks):
    out = np.full(n_blocks, np.nan)
    order = np.argsort(blocks[mask], kind="stable")
    b = blocks[mask][order]
    v = values[mask][order]
    uniq, start = np.unique(b, return_index=True)
    sums = np.add.reduceat(v, start)
    counts = np.diff(np.append(start, len(v)))
    out[uniq] = sums / counts
    return out


def block_table(oof, none, d):
    """Per-block means of every quantity the estimands are built from."""
    nb = d.n_blocks
    t = {
        "eval": _block_means(oof, d.cued_block, d.cued_y == 1, nb),
        "deploy": _block_means(oof, d.cued_block, d.cued_y == 0, nb),
        "none_bench": _block_means(none, d.none_block, d.none_fmt == 1, nb),
        "none_casual": _block_means(none, d.none_block, d.none_fmt == 0, nb),
    }
    for t_i, name in enumerate(d.tpl_names):
        t[f"none_tpl::{name}"] = _block_means(none, d.none_block, d.none_tpl == t_i, nb)
    return t


def estimands_from_blocks(t, keys=None):
    """The point estimates, given a (possibly resampled) block table."""
    mE, mD = np.nanmean(t["eval"]), np.nanmean(t["deploy"])
    d_stated = mE - mD
    mB, mC = np.nanmean(t["none_bench"]), np.nanmean(t["none_casual"])
    d_format = np.nanmean(t["none_bench"] - t["none_casual"])  # paired within block

    out = {
        "delta_stated": d_stated,
        "delta_format": d_format,
        "rho": d_format / d_stated,
        "pi_bench": (mB - mD) / d_stated,
        "pi_casual": (mC - mD) / d_stated,
        "pi_none": ((mB + mC) / 2 - mD) / d_stated,
    }
    for k in (keys or []):
        if k.startswith("none_tpl::"):
            out["pi_" + k] = (np.nanmean(t[k]) - mD) / d_stated
    return out


def paired_stats(t):
    """Standardized effect size and AUC for the no-cue format shift.

    Endpoint E6 of the frozen plan, computed block-paired: `smd` is the
    within-block benchmark-minus-casual difference over its own SD, and `auc`
    is the probability that a block's benchmark mean exceeds its casual mean
    (a paired, block-level readout, not an item-level one).
    """
    diff = t["none_bench"] - t["none_casual"]
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return {"smd_paired": float(diff.mean() / sd) if sd > 0 else np.nan,
            "block_win_rate": float(wins / len(diff)),
            "n_blocks": int(len(diff))}


def block_bootstrap(t, n_boot=10000, seed=0, keys=None):
    """Cluster bootstrap over payload blocks; percentile 95% CIs.

    Blocks are resampled once per draw and every estimand is recomputed from
    the same resample, so ratios such as `rho` inherit the correlation between
    their numerator and denominator instead of being combined independently.
    """
    nb = len(t["eval"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))

    diff = t["none_bench"] - t["none_casual"]
    wins = np.where(np.isnan(diff), np.nan, (diff > 0).astype(float) + 0.5 * (diff == 0))

    def bmean(v):
        return np.nanmean(v[idx], axis=1)

    with np.errstate(invalid="ignore"):
        mE, mD = bmean(t["eval"]), bmean(t["deploy"])
        d_stated = mE - mD
        d_format = bmean(diff)
        draws = {
            "delta_stated": d_stated,
            "delta_format": d_format,
            "rho": d_format / d_stated,
            "pi_bench": (bmean(t["none_bench"]) - mD) / d_stated,
            "pi_casual": (bmean(t["none_casual"]) - mD) / d_stated,
            "smd_paired": d_format / np.nanstd(diff[idx], axis=1, ddof=1),
            "block_win_rate": bmean(wins),
        }
        for k in (keys or []):
            if k.startswith("none_tpl::"):
                draws["pi_" + k] = (bmean(t[k]) - mD) / d_stated

    out = {}
    for k, col in draws.items():
        col = col[np.isfinite(col)]
        out[k] = {"ci95": [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))],
                  "boot_mean": float(col.mean()), "n_valid": int(len(col))}
    return out
