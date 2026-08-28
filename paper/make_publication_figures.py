"""Generate the uniform publication figure suite for the rewritten TAE paper.

All numerical panels are derived from locked local artifacts. The script performs
contract checks before plotting and writes PNG, PDF, and SVG versions plus a
hash-stamped source/output manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os

from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures_rewrite"

ESTIMATOR_PATH = ROOT / "artifacts" / "exploratory" / "full320_exact_logistic_estimator_reversal_v1_20260823" / "estimator_reversal_results.json"
TRANSFER_PATH = ROOT / "artifacts" / "exploratory" / "full320_orthc_transfer_positive_20260826" / "transfer_results.json"
TRANSFER_CSV = ROOT / "artifacts" / "exploratory" / "full320_orthc_transfer_positive_20260826" / "payload_block_effects.csv"
EXTENSION_PATH = ROOT / "artifacts" / "exploratory" / "full320_orthc_llama31_70b_model_extension_positive_20260827" / "extension_results.json"
EXTENSION_CSV = ROOT / "artifacts" / "exploratory" / "full320_orthc_llama31_70b_model_extension_positive_20260827" / "payload_block_effects.csv"
INVALID_GATE_PATH = ROOT / "artifacts" / "exploratory" / "pilot80_factorial_causal_v2_invalid_20260823" / "run_manifest.json"
EQUAL_N_PATH = ROOT / "artifacts" / "full320_equal_n_llama31_8b_l12.json"
JLENS_PATH = ROOT / "runs" / "tae_2026" / "jlens_v3" / "20260818_full320_8b_retry03" / "outputs" / "assay.json"
PINNED_DIR = ROOT / "artifacts" / "pinned_full320"
BOOTSTRAP_PATH = ROOT / "artifacts" / "format_bound_regrowth_bootstrap.json"
CAUSAL_PATH = Path(
    os.environ.get(
        "TAE_PILOT_CAUSAL_RESULTS",
        r"C:/format-EA-data/pilot80_v3_complete_evidence/pilot80_factorial_causal_v3_analysis/causal_results.json",
    )
)

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
}
MODEL_ORDER = ["Llama-3.1-8B", "Llama-3.3-70B", "Llama-3.1-70B"]
MODEL_COLORS = {
    "Llama-3.1-8B": "#0072B2",
    "Llama-3.3-70B": "#D55E00",
    "Llama-3.1-70B": "#009E73",
}
COLORS = {
    "ink": "#202124",
    "muted": "#666A70",
    "grid": "#D9DDE3",
    "light": "#F4F6F8",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#6F4E9C",
    "sky": "#56B4E9",
    "pass": "#16825D",
    "fail": "#B23A48",
    "neutral": "#7A7F87",
}
ESTIMATOR_COLORS = {"diagonal_lda": COLORS["purple"], "logistic": COLORS["vermillion"], "interaction": COLORS["sky"]}
ESTIMATOR_LABELS = {"diagonal_lda": "Diagonal LDA", "logistic": "L2 logistic", "interaction": "Interaction pipeline"}

SOURCE_FILES = [
    ESTIMATOR_PATH,
    TRANSFER_PATH,
    TRANSFER_CSV,
    EXTENSION_PATH,
    EXTENSION_CSV,
    INVALID_GATE_PATH,
    EQUAL_N_PATH,
    JLENS_PATH,
    CAUSAL_PATH,
    PINNED_DIR / "layer_sweep_full320-v3_llama31_8b.json",
    PINNED_DIR / "layer_sweep_full320-v3_llama31_70b.json",
    PINNED_DIR / "layer_sweep_full320-v3_llama33_70b.json",
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def clean_axis(ax: mpl.axes.Axes, *, grid: str | None = "y") -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left", color=COLORS["ink"])


def signed_formatter(x: float, _: int) -> str:
    if abs(x) < 5e-7:
        return "0"
    return f"{x:+.3f}"


def save_figure(
    fig: mpl.figure.Figure,
    stem: str,
    *,
    sources: Iterable[Path],
    description: str,
    manifest: dict[str, Any],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    outputs: dict[str, dict[str, Any]] = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(path, **kwargs)
        outputs[suffix] = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest[stem] = {
        "description": description,
        "sources": [{"path": rel(path), "sha256": sha256_file(path)} for path in sources],
        "outputs": outputs,
        "size_inches": [float(v) for v in fig.get_size_inches()],
    }
    plt.close(fig)


def errorbarh(ax: mpl.axes.Axes, y: float, point: float, low: float, high: float, color: str, *, marker: str = "o", zorder: int = 4) -> None:
    ax.errorbar(
        point,
        y,
        xerr=np.array([[point - low], [high - point]]),
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        elinewidth=1.2,
        capsize=2.3,
        capthick=1.0,
        zorder=zorder,
    )


def bootstrap_ci(values: np.ndarray, *, seed: int = 20260827, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def primary_effects(rows: list[dict[str, str]], model_id: str | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if model_id is not None and row.get("model_id") != model_id:
            continue
        if row["family"] == "C_high_minus_low" and row["alpha"] == "1.0" and row["depth"] == "high_minus_low" and row["endpoint"] == "interaction" and not row.get("control_seed"):
            out[row["payload_block_id"]] = float(row["point"])
    return out


def endpoint_effects(rows: list[dict[str, str]], model_id: str, endpoint: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if row.get("model_id") != model_id:
            continue
        if row["family"] == "C_high_minus_low" and row["alpha"] == "1.0" and row["depth"] == "high_minus_low" and row["endpoint"] == endpoint and not row.get("control_seed"):
            out[row["payload_block_id"]] = float(row["point"])
    return out


def cross_format_curves(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    n_layers = len(rows)
    depth = np.asarray([float(row["layer"]) / (n_layers - 1) for row in rows])
    dual_format = np.asarray([float(row["decorrelated"]["auc"]) for row in rows])
    benchmark_to_casual = np.asarray([float(row["b2c"]["auc"]) for row in rows])
    casual_to_benchmark = np.asarray([float(row["c2b"]["auc"]) for row in rows])
    cross_format_mean = (benchmark_to_casual + casual_to_benchmark) / 2
    return {
        "depth": depth,
        "dual_format": dual_format,
        "benchmark_to_casual": benchmark_to_casual,
        "casual_to_benchmark": casual_to_benchmark,
        "cross_format_mean": cross_format_mean,
        "gap": dual_format - cross_format_mean,
    }


def cross_format_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    curves = cross_format_curves(rows)
    depth = curves["depth"]
    dual_format = curves["dual_format"]
    cross_format = curves["cross_format_mean"]
    gap = curves["gap"]
    late = depth >= 0.5
    peak_index = int(np.argmax(dual_format))
    maximum_index = int(np.argmax(gap))
    audit_index = int(np.argmin(np.abs(depth - 0.375)))
    dual_slope = float(np.polyfit(depth[late], dual_format[late], 1)[0])
    cross_slope = float(np.polyfit(depth[late], cross_format[late], 1)[0])
    return {
        "n_layers": len(rows),
        "all_layer_gaps_positive": bool(np.all(gap > 0)),
        "dual_format_peak": {
            "layer": int(rows[peak_index]["layer"]),
            "normalized_depth": float(depth[peak_index]),
            "auc": float(dual_format[peak_index]),
            "cross_format_mean_auc": float(cross_format[peak_index]),
            "gap": float(gap[peak_index]),
        },
        "maximum_gap": {
            "layer": int(rows[maximum_index]["layer"]),
            "normalized_depth": float(depth[maximum_index]),
            "gap": float(gap[maximum_index]),
        },
        "q_0.375_nearest": {
            "layer": int(rows[audit_index]["layer"]),
            "normalized_depth": float(depth[audit_index]),
            "dual_format_auc": float(dual_format[audit_index]),
            "cross_format_mean_auc": float(cross_format[audit_index]),
            "gap": float(gap[audit_index]),
        },
        "late_half_mean_gap": float(np.mean(gap[late])),
        "late_half_dual_format_slope_per_unit_depth": dual_slope,
        "late_half_cross_format_slope_per_unit_depth": cross_slope,
        "late_half_gap_slope_per_unit_depth": float(dual_slope - cross_slope),
        "final_layer_gap": float(gap[-1]),
        "mean_absolute_direction_difference": float(
            np.mean(np.abs(curves["benchmark_to_casual"] - curves["casual_to_benchmark"]))
        ),
    }


def validate_contracts(data: dict[str, Any]) -> dict[str, Any]:
    estimator = data["estimator"]
    causal = data["causal"]
    transfer = data["transfer"]
    extension = data["extension"]
    invalid = data["invalid"]
    transfer_rows = data["transfer_rows"]
    extension_rows = data["extension_rows"]

    assert estimator["status"] == "completed_targeted_estimator_reversal"
    assert estimator["point_summary"]["aggregate"]["diagonal_lda"]["macro_format_roc_auc"] < 0
    assert estimator["point_summary"]["aggregate"]["logistic"]["macro_format_roc_auc"] > 0
    assert causal["decision"] == "negative"
    assert causal["criteria"]["C_interaction"] is True
    assert causal["criteria"]["A_shared"] is False and causal["criteria"]["B_format"] is False
    assert causal["criteria"]["endpoint_valid"] is False
    assert transfer["decision"] == "positive_replication"
    assert extension["decision"] == "positive_model_extension"
    assert invalid["use_for_inference"] is False
    assert invalid["partial_output_policy"]["status"] == "excluded_wholesale"
    cross_format = {
        label: cross_format_summary(rows) for label, rows in data["sweeps"].items()
    }
    for label, summary in cross_format.items():
        assert summary["all_layer_gaps_positive"], label
        assert summary["late_half_gap_slope_per_unit_depth"] > 0, label
        assert (
            summary["late_half_cross_format_slope_per_unit_depth"]
            < summary["late_half_dual_format_slope_per_unit_depth"]
        ), label
    equal_n = data["equal_n"]["results"]
    equal_n_mixed_min = min(float(row["equal_n_mixed"]["auc"]) for row in equal_n)
    equal_n_single_max = max(
        max(float(row[key]["auc"]) for key in ("benchmark_only", "casual_only"))
        for row in equal_n
    )
    assert equal_n_mixed_min > equal_n_single_max

    model_counts: dict[str, int] = {}
    for model_id in transfer["models"]:
        effects = primary_effects(transfer_rows, model_id)
        assert len(effects) == 80 and min(effects.values()) > 0
        model_counts[MODEL_LABELS[model_id]] = len(effects)
        expected = transfer["models"][model_id]["primary_high_minus_low"]["mean"]
        assert math.isclose(float(np.mean(list(effects.values()))), expected, rel_tol=0, abs_tol=2e-12)
    extension_effects = primary_effects(extension_rows)
    assert len(extension_effects) == 80 and min(extension_effects.values()) > 0
    assert math.isclose(float(np.mean(list(extension_effects.values()))), extension["primary_high_minus_low"]["mean"], rel_tol=0, abs_tol=2e-12)
    model_counts["Llama-3.1-70B"] = len(extension_effects)

    assert causal["reproduction"]["source_max_abs"] == 0.0
    assert causal["reproduction"]["source_min_cosine"] >= 0.999
    assert len(causal["response_metrics"]) == 72
    for model_id, metrics in causal["endpoint_metrics"].items():
        for endpoint, values in metrics.items():
            assert 0 <= values["roc_auc"] <= 1, (model_id, endpoint)
    return {
        "validated": True,
        "transfer_payload_blocks": model_counts,
        "causal_response_metrics": len(causal["response_metrics"]),
        "source_files": len(SOURCE_FILES),
        "cross_format_generalization_gap": cross_format,
        "format_gap_regrowth": format_gap_regrowth_summary(data["sweeps"]),
        "equal_n_8b_control": {
            "mixed_min_auc": equal_n_mixed_min,
            "single_format_max_auc": equal_n_single_max,
            "minimum_separation": equal_n_mixed_min - equal_n_single_max,
        },
    }


def figure_design_overview(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(7.0, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.28], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 2.9)
    ax.set_ylim(0, 2.55)
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Matched 2 × 2 design", loc="left", pad=8)

    x0, y0, w, h = 0.82, 0.28, 0.92, 0.78
    col_names = ["Benchmark\nformat", "Casual\nformat"]
    row_names = ["Evaluation\npurpose", "Deployment\npurpose"]
    cell_fills = [["#E9F3FA", "#EAF7F2"], ["#F7F0E4", "#F6ECED"]]
    for c, name in enumerate(col_names):
        ax.text(x0 + c * w + w / 2, y0 + 2 * h + 0.23, name, ha="center", va="center", fontsize=8, fontweight="bold")
    for r, name in enumerate(row_names):
        ax.text(x0 - 0.12, y0 + (1 - r) * h + h / 2, name, ha="right", va="center", fontsize=8, fontweight="bold")
    labels = [["E × benchmark", "E × casual"], ["D × benchmark", "D × casual"]]
    for r in range(2):
        for c in range(2):
            yy = y0 + (1 - r) * h
            rect = FancyBboxPatch((x0 + c * w, yy), w - 0.04, h - 0.04, boxstyle="round,pad=0.02,rounding_size=0.035", facecolor=cell_fills[r][c], edgecolor="#AEB5BE", linewidth=0.85)
            ax.add_patch(rect)
            ax.text(x0 + c * w + (w - 0.04) / 2, yy + (h - 0.04) / 2, labels[r][c], ha="center", va="center", fontsize=7.6)

    ax.text(0.03, 0.13, "Same payload appears in every cell", fontsize=7.3, color=COLORS["muted"], ha="left")
    ax.text(0.03, 0.01, "Purpose and format vary independently", fontsize=7.3, color=COLORS["muted"], ha="left")

    bx = fig.add_subplot(gs[0, 1])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    panel_label(bx, "B")
    bx.set_title("One question, progressively stronger evidence", loc="left", pad=8)
    stages = [
        (0.81, "1", "Factorial geometry", "What is linearly decodable?", COLORS["blue"]),
        (0.59, "2", "Estimator audit", "Does the conclusion survive a probe change?", COLORS["purple"]),
        (0.37, "3", "Frozen intervention", "Which components propagate causally?", COLORS["vermillion"]),
        (0.15, "4", "Prospective transfer", "Does the fixed response reproduce on new payloads?", COLORS["green"]),
    ]
    for idx, (y, number, title, subtitle, color) in enumerate(stages):
        box = FancyBboxPatch((0.06, y - 0.075), 0.87, 0.145, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor="white", edgecolor="#B6BCC4", linewidth=0.8)
        bx.add_patch(box)
        bx.add_patch(FancyBboxPatch((0.083, y - 0.036), 0.075, 0.075, boxstyle="circle,pad=0.0", facecolor=color, edgecolor=color))
        bx.text(0.121, y, number, color="white", fontsize=8, fontweight="bold", ha="center", va="center")
        bx.text(0.19, y + 0.025, title, fontsize=8.2, fontweight="bold", ha="left", va="center")
        bx.text(0.19, y - 0.027, subtitle, fontsize=7.1, color=COLORS["muted"], ha="left", va="center")
        if idx < len(stages) - 1:
            bx.add_patch(FancyArrowPatch((0.495, y - 0.08), (0.495, stages[idx + 1][0] + 0.082), arrowstyle="-|>", mutation_scale=8, color="#A2A8B0", linewidth=0.8))
    bx.text(0.5, 0.005, "Claim boundary: internal stated-purpose response, not evaluation awareness or behavior", ha="center", va="bottom", fontsize=7.1, color=COLORS["muted"])

    save_figure(fig, "main_01_design_overview", sources=[], description="Matched purpose-by-format design and evidence ladder.", manifest=manifest)


def figure_cross_format_gap(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    sweeps = data["sweeps"]
    fig = plt.figure(figsize=(7.0, 4.7), constrained_layout=True)
    fig.suptitle(
        "Probe trained on both formats (solid) vs one-to-other format transfer (dashed)",
        fontsize=9.5,
        fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.88], hspace=0.22, wspace=0.18)
    top_axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    summaries: dict[str, dict[str, Any]] = {}

    for index, (ax, label) in enumerate(zip(top_axes, MODEL_ORDER)):
        curves = cross_format_curves(sweeps[label])
        summaries[label] = cross_format_summary(sweeps[label])
        depth_percent = 100 * curves["depth"]
        color = MODEL_COLORS[label]
        panel_label(ax, chr(ord("A") + index))
        ax.set_title(label, loc="left")
        ax.fill_between(
            depth_percent,
            curves["cross_format_mean"],
            curves["dual_format"],
            color=mpl.colors.to_rgba(color, 0.14),
            linewidth=0,
            zorder=1,
        )
        ax.plot(depth_percent, curves["dual_format"], color=color, linewidth=1.7, zorder=3)
        ax.plot(
            depth_percent,
            curves["cross_format_mean"],
            color=color,
            linewidth=1.7,
            linestyle=(0, (4, 2.5)),
            zorder=3,
        )
        ax.axhline(0.5, color=COLORS["neutral"], linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.48, 1.015)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Relative depth (%)")
        clean_axis(ax, grid="y")
        ax.text(
            0.97,
            0.08,
            f"late-half mean $\\Delta$={summaries[label]['late_half_mean_gap']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.7,
            color=COLORS["muted"],
        )
    top_axes[0].set_ylabel("Cue-family-held-out ROC AUC")

    ax = fig.add_subplot(gs[1, :])
    ax.text(-0.055, 1.18, "D", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left", color=COLORS["ink"])
    ax.set_title("Cross-format generalization gap", loc="left")
    ax.axvspan(50, 100, color=mpl.colors.to_rgba(COLORS["neutral"], 0.07), zorder=0)
    ax.text(51.5, 0.012, "late half", color=COLORS["muted"], fontsize=6.8, va="bottom")
    for label in MODEL_ORDER:
        curves = cross_format_curves(sweeps[label])
        summary = summaries[label]
        ax.plot(
            100 * curves["depth"],
            curves["gap"],
            color=MODEL_COLORS[label],
            linewidth=1.7,
            label=f"{label}  late mean {summary['late_half_mean_gap']:.3f}",
        )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.006, 0.235)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative depth through the network (%)")
    ax.set_ylabel("AUC gap (both-format probe − cross mean)")
    clean_axis(ax, grid="y")
    ax.legend(loc="upper right", ncol=1)
    fig.text(
        0.5,
        -0.035,
        "Held cue-vocabulary-family OOF · L2 logistic · 7,680 stated rows / 320 payload blocks.\n"
        "Descriptive generalization contrast; not a decomposition of representation content.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    save_figure(
        fig,
        "main_02_cross_format_generalization",
        sources=list(data["sweep_paths"].values()),
        description=(
            "Held cue-vocabulary-family both-format probe decoding versus the mean of both "
            "one-to-other format transfer directions across normalized depth, with the descriptive AUC gap."
        ),
        manifest=manifest,
    )


def format_gap_regrowth_summary(sweeps: dict[str, Any]) -> dict[str, Any]:
    boot = load_json(BOOTSTRAP_PATH) if BOOTSTRAP_PATH.exists() else {}
    boot_key = {"Llama-3.1-8B": "llama31_8b", "Llama-3.1-70B": "llama31_70b", "Llama-3.3-70B": "llama33_70b"}
    out: dict[str, Any] = {}
    for label in MODEL_ORDER:
        curves = cross_format_curves(sweeps[label])
        gap = curves["gap"]
        min_index = int(np.argmin(gap))
        out[label] = {
            "minimum_depth_pct": float(100 * curves["depth"][min_index]),
            "gap_at_minimum": float(gap[min_index]),
            "gap_at_final_layer": float(gap[-1]),
            "final_over_minimum_ratio": float(gap[-1] / gap[min_index]),
            "bootstrap": boot.get(boot_key[label]),
        }
    return out


def figure_format_gap_regrowth(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    sweeps = data["sweeps"]
    summary = format_gap_regrowth_summary(sweeps)
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.axvspan(0, 12, color=mpl.colors.to_rgba(COLORS["neutral"], 0.06), zorder=0)
    ax.text(1, 0.225, "early spike\n(format-entangled\nlexical cues)", fontsize=6.6, color=COLORS["muted"], va="top")
    ax.axvspan(50, 100, color=mpl.colors.to_rgba(COLORS["neutral"], 0.06), zorder=0)
    ax.text(51.5, 0.225, "late half", fontsize=6.8, color=COLORS["muted"], va="top")

    for label in MODEL_ORDER:
        curves = cross_format_curves(sweeps[label])
        depth_pct = 100 * curves["depth"]
        color = MODEL_COLORS[label]
        ax.plot(depth_pct, curves["gap"], color=color, linewidth=1.7, zorder=3, label=label)
        s = summary[label]
        ax.scatter([s["minimum_depth_pct"]], [s["gap_at_minimum"]], color=color, s=22, zorder=4, edgecolor="white", linewidth=0.5)
        ax.scatter([100.0], [s["gap_at_final_layer"]], color=color, s=22, zorder=4, marker="D", edgecolor="white", linewidth=0.5)

    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.006, 0.235)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative depth through the network (%)")
    ax.set_ylabel("AUC gap (both-format probe \u2212 cross-format transfer mean)")
    clean_axis(ax, grid="y")
    ax.set_title(
        "Cross-format generalization gap is non-monotonic: it collapses near a\n"
        "format-invariant minimum, then re-widens through the late layers",
        loc="left", fontsize=9.0,
    )
    handles, _ = ax.get_legend_handles_labels()
    marker_handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS["muted"], markersize=5, label="per-model minimum"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["muted"], markersize=5, label="final layer"),
    ]
    ax.legend(handles=handles + marker_handles, loc="upper right", ncol=1, fontsize=7.0)

    footer_lines = [
        "Held cue-vocabulary-family OOF \u00b7 L2 logistic \u00b7 7,680 stated rows / 320 payload blocks "
        "(3,840 no-cue control rows excluded; no purpose label).",
    ]
    boot_parts = [
        f"{label} widening 95% CI [{summary[label]['bootstrap']['widening_ci95'][0]:.3f}, "
        f"{summary[label]['bootstrap']['widening_ci95'][1]:.3f}]"
        for label in MODEL_ORDER if summary[label]["bootstrap"]
    ]
    if boot_parts:
        footer_lines.append(
            "Late-layer re-widening (final-layer gap \u2212 minimum-layer gap), 1,000-rep payload-block bootstrap:"
        )
        footer_lines.append("; ".join(boot_parts) + ".")
    footer_lines.append("Descriptive generalization contrast; not a decomposition of representation content or a causal claim.")
    fig.text(0.5, -0.05 - 0.028 * (len(footer_lines) - 1), "\n".join(footer_lines), ha="center", va="top", fontsize=6.6, color=COLORS["muted"])

    save_figure(
        fig,
        "main_02b_format_gap_regrowth",
        sources=list(data["sweep_paths"].values()) + ([BOOTSTRAP_PATH] if BOOTSTRAP_PATH.exists() else []),
        description=(
            "Cross-format generalization gap annotated with each model's mid-depth minimum and final-layer "
            "value, quantifying the late-layer re-widening (with payload-block bootstrap CIs where available)."
        ),
        manifest=manifest,
    )

def figure_estimator_reversal(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    result = data["estimator"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), gridspec_kw={"width_ratios": [0.88, 1.25]}, constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("Aggregate contrast", loc="left")
    metrics = [("macro_format_roc_auc", "AUC"), ("macro_format_balanced_accuracy", "Balanced accuracy")]
    offsets = {"diagonal_lda": -0.13, "logistic": 0.13}
    for base_y, (metric, label) in enumerate(metrics[::-1]):
        for estimator_name in ("diagonal_lda", "logistic"):
            point = result["point_summary"]["aggregate"][estimator_name][metric]
            ci = result["bootstrap"]["aggregate"][estimator_name][metric]
            y = base_y + offsets[estimator_name]
            errorbarh(ax, y, point, ci["ci_2.5"], ci["ci_97.5"], ESTIMATOR_COLORS[estimator_name], marker="o" if estimator_name == "logistic" else "s")
    ax.axvline(0, color=COLORS["ink"], linewidth=0.9, zorder=1)
    ax.set_yticks([0, 1], ["Balanced accuracy", "AUC"])
    ax.set_xlim(-0.012, 0.034)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(signed_formatter))
    ax.set_xlabel("Mixed − best specialist")
    clean_axis(ax, grid="x")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", color=ESTIMATOR_COLORS["diagonal_lda"], label="Diagonal LDA"),
            Line2D([], [], marker="o", linestyle="none", color=ESTIMATOR_COLORS["logistic"], label="L2 logistic"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
    )

    panel_label(bx, "B")
    bx.set_title("AUC contrast by checkpoint", loc="left")
    model_ids = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]
    ybase = np.arange(len(model_ids))[::-1]
    for y, model_id in zip(ybase, model_ids):
        for estimator_name in ("diagonal_lda", "logistic"):
            point = result["point_summary"]["by_model"][model_id][estimator_name]["macro_format_roc_auc"]
            ci = result["bootstrap"]["by_model"][model_id][estimator_name]["macro_format_roc_auc"]
            yy = y + offsets[estimator_name]
            errorbarh(bx, yy, point, ci["ci_2.5"], ci["ci_97.5"], ESTIMATOR_COLORS[estimator_name], marker="o" if estimator_name == "logistic" else "s")
    bx.axvline(0, color=COLORS["ink"], linewidth=0.9, zorder=1)
    bx.set_yticks(ybase, [MODEL_LABELS[m] for m in model_ids])
    bx.set_xlim(-0.012, 0.020)
    bx.xaxis.set_major_locator(MultipleLocator(0.005))
    bx.xaxis.set_major_formatter(FuncFormatter(signed_formatter))
    bx.set_xlabel("Mixed − best specialist AUC")
    clean_axis(bx, grid="x")
    bx.text(0.99, -0.25, "95% payload-block bootstrap intervals", transform=bx.transAxes, ha="right", va="top", fontsize=7, color=COLORS["muted"])

    save_figure(fig, "main_03_estimator_reversal", sources=[ESTIMATOR_PATH], description="Exact-row estimator reversal in the mixed-minus-best-specialist observational contrast.", manifest=manifest)


def figure_causal_decision(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    causal = data["causal"]
    fig = plt.figure(figsize=(7.0, 3.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.05, 0.95], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[0, 2])

    panel_label(ax, "A")
    ax.set_title("Baseline endpoint gate", loc="left")
    endpoints = ["purpose_benchmark", "purpose_casual", "format"]
    endpoint_labels = ["Purpose\nbenchmark", "Purpose\ncasual", "Format"]
    model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    offsets = [-0.11, 0.11]
    for i, endpoint in enumerate(endpoints):
        for offset, model_id in zip(offsets, model_ids):
            values = causal["endpoint_metrics"][model_id][endpoint]
            label = MODEL_LABELS[model_id]
            marker = "o" if values["pass"] else "X"
            ax.scatter(i + offset, values["roc_auc"], s=42 if values["pass"] else 58, marker=marker, color=MODEL_COLORS[label] if values["pass"] else COLORS["fail"], edgecolor="white", linewidth=0.6, zorder=4)
            if not values["pass"]:
                ax.annotate("FAIL", (i + offset, values["roc_auc"]), xytext=(7, -1), textcoords="offset points", ha="left", va="center", fontsize=6.8, fontweight="bold", color=COLORS["fail"])
    ax.axhline(0.8, color=COLORS["fail"], linestyle="--", linewidth=1.0)
    ax.text(2.48, 0.804, "frozen gate", fontsize=6.8, color=COLORS["fail"], va="bottom", ha="right")
    ax.set_xticks(range(3), endpoint_labels)
    ax.set_ylim(0.74, 1.015)
    ax.set_ylabel("Baseline ROC AUC")
    clean_axis(ax, grid="y")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=MODEL_COLORS[MODEL_LABELS[m]], label=MODEL_LABELS[m]) for m in model_ids], loc="upper left")

    panel_label(bx, "B")
    bx.set_title("Frozen decision criteria", loc="left")
    criteria_order = [
        ("hook_and_reproduction", "Reproduction"),
        ("endpoint_valid", "Endpoint"),
        ("A_shared", "$A$ shared"),
        ("B_format", "$B$ format"),
        ("C_interaction", "$C$ interaction"),
        ("dose_response", "Dose"),
        ("factorial_cube_agreement", "Additivity"),
        ("geometry_to_causality", "High − low"),
    ]
    bx.set_xlim(0, 1)
    bx.set_ylim(-0.65, len(criteria_order) - 0.35)
    bx.axis("off")
    for y, (key, label) in enumerate(criteria_order[::-1]):
        passed = bool(causal["criteria"][key])
        color = COLORS["pass"] if passed else COLORS["fail"]
        bx.text(0.02, y, label, ha="left", va="center", fontsize=7.4)
        box = FancyBboxPatch((0.64, y - 0.28), 0.31, 0.56, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=mpl.colors.to_rgba(color, 0.10), edgecolor=color, linewidth=0.85)
        bx.add_patch(box)
        bx.text(0.795, y, "PASS" if passed else "FAIL", ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
    bx.text(0.5, -0.58, "Overall frozen decision: NEGATIVE", ha="center", va="center", fontsize=7.7, fontweight="bold", color=COLORS["fail"])

    panel_label(cx, "C")
    cx.set_title("Predeclared high-layer $C$", loc="left")
    intended = {"A": "shared_purpose", "B": "format", "C": "purpose_interaction"}
    high_layer = {"meta-llama/Llama-3.1-8B-Instruct": 29, "meta-llama/Llama-3.3-70B-Instruct": 74}
    rows = [r for r in causal["response_metrics"] if r["alpha"] == 1.0 and r["component"] == "C" and r["endpoint"] == intended["C"] and r["source_layer"] == high_layer[r["model_id"]]]
    rows.sort(key=lambda r: MODEL_ORDER.index(MODEL_LABELS[r["model_id"]]))
    ypos = np.arange(len(rows))[::-1]
    for y, row in zip(ypos, rows):
        label = MODEL_LABELS[row["model_id"]]
        errorbarh(cx, y, row["point"], row["ci_2.5"], row["ci_97.5"], MODEL_COLORS[label])
        cx.text(row["ci_97.5"] + 0.025, y, f"{row['point']:.3f}", va="center", fontsize=7, color=COLORS["ink"])
    cx.axvline(0, color=COLORS["ink"], linewidth=0.8)
    cx.set_yticks(ypos, [MODEL_LABELS[r["model_id"]] for r in rows])
    cx.set_xlim(-0.03, 0.89)
    cx.set_xlabel("Purpose-interaction readout effect")
    clean_axis(cx, grid="x")
    cx.text(0.98, 0.48, "$p_{random}=1/33$ in both", transform=cx.transAxes, ha="right", va="center", fontsize=7, color=COLORS["muted"])

    save_figure(fig, "main_04_frozen_causal_decision", sources=[CAUSAL_PATH], description="Valid baseline endpoint gate, complete frozen criterion decision, and the narrower passing high-layer C criterion.", manifest=manifest)


def figure_causal_linearity(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    causal = data["causal"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.2, 0.8]}, constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("High-layer $C$ response scales with dose", loc="left")
    high_layer = {"meta-llama/Llama-3.1-8B-Instruct": 29, "meta-llama/Llama-3.3-70B-Instruct": 74}
    for model_id, layer in high_layer.items():
        label = MODEL_LABELS[model_id]
        rows = [r for r in causal["response_metrics"] if r["model_id"] == model_id and r["source_layer"] == layer and r["component"] == "C" and r["endpoint"] == "purpose_interaction"]
        rows.sort(key=lambda r: r["alpha"])
        xs = np.array([0.0] + [r["alpha"] for r in rows])
        ys = np.array([0.0] + [r["point"] for r in rows])
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[label], label=label, zorder=3)
        for row in rows:
            ax.errorbar(row["alpha"], row["point"], yerr=np.array([[row["point"] - row["ci_2.5"]], [row["ci_97.5"] - row["point"]]]), fmt="none", ecolor=MODEL_COLORS[label], elinewidth=1.0, capsize=2.2, zorder=2)
        ax.plot([0, 1], [0, rows[-1]["point"]], color=MODEL_COLORS[label], linestyle="--", alpha=0.38, linewidth=1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Intervention strength $\\alpha$")
    ax.set_ylabel("Purpose-interaction readout effect")
    ax.set_ylim(-0.02, 0.84)
    clean_axis(ax, grid="y")
    ax.legend(loc="upper left")

    panel_label(bx, "B")
    bx.set_title("Eight-corner additivity", loc="left")
    model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    y = np.arange(2)[::-1]
    bx.axvspan(0.5, 1.0, color=mpl.colors.to_rgba(COLORS["pass"], 0.07), zorder=0)
    bx.axvline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    for yy, model_id in zip(y, model_ids):
        label = MODEL_LABELS[model_id]
        values = causal["factorial_cube"][model_id]
        bx.scatter(values["additive_r2"], yy, s=50, color=MODEL_COLORS[label], edgecolor="white", linewidth=0.6, zorder=4)
        bx.text(values["additive_r2"] - 0.012, yy + 0.17, f"$R^2$={values['additive_r2']:.4f}", ha="right", va="center", fontsize=7, color=MODEL_COLORS[label])
        bx.text(0.515, yy - 0.12, f"RMSE={values['rmse']:.3f}", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])
    bx.set_yticks(y, [MODEL_LABELS[m] for m in model_ids])
    bx.set_ylim(-0.42, 1.42)
    bx.set_xlim(0.45, 1.015)
    bx.set_xlabel("Additive prediction $R^2$")
    clean_axis(bx, grid="x")
    bx.text(0.515, 1.26, "frozen threshold", fontsize=6.7, color=COLORS["neutral"], ha="left", va="center")

    save_figure(fig, "main_05_causal_linearity", sources=[CAUSAL_PATH], description="Dose linearity and additive agreement of the full factorial intervention cube.", manifest=manifest)


def transfer_model_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    transfer = data["transfer"]
    extension = data["extension"]
    rows: list[dict[str, Any]] = []
    for model_id in ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"):
        model = transfer["models"][model_id]
        rows.append(
            {
                "model_id": model_id,
                "label": MODEL_LABELS[model_id],
                "mean": model["primary_high_minus_low"]["mean"],
                "ci": model["primary_high_minus_low"]["ci95_payload_bootstrap"],
                "half": model["half_alpha_high_minus_low"]["mean"],
                "randoms": model["random_control_calibration"]["high_minus_low_means"],
                "random_threshold": model["random_control_calibration"]["absolute_mean_97_5_percentile"],
                "extension": False,
            }
        )
    rows.append(
        {
            "model_id": extension["model_id"],
            "label": MODEL_LABELS[extension["model_id"]],
            "mean": extension["primary_high_minus_low"]["mean"],
            "ci": extension["primary_high_minus_low"]["ci95_payload_bootstrap"],
            "half": extension["half_alpha_high_minus_low"]["mean"],
            "randoms": extension["random_control_calibration"]["high_minus_low_means"],
            "random_threshold": extension["random_control_calibration"]["absolute_mean_97_5_percentile"],
            "extension": True,
        }
    )
    return rows


def figure_prospective_transfer(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    rows = transfer_model_data(data)
    fig = plt.figure(figsize=(7.0, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.9, 1.15], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[0, 2])

    y = np.arange(len(rows))[::-1]
    panel_label(ax, "A")
    ax.set_title("Prospective high − low transfer", loc="left")
    for yy, row in zip(y, rows):
        marker = "D" if row["extension"] else "o"
        errorbarh(ax, yy, row["mean"], row["ci"][0], row["ci"][1], MODEL_COLORS[row["label"]], marker=marker)
        ax.text(row["ci"][1] + 0.017, yy, f"{row['mean']:.3f}", va="center", fontsize=7, color=COLORS["ink"])
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_yticks(y, [r["label"] + ("  extension" if r["extension"] else "") + "\n80/80 positive" for r in rows])
    ax.set_xlim(-0.03, 0.82)
    ax.set_xlabel("Purpose-interaction readout effect")
    clean_axis(ax, grid="x")

    panel_label(bx, "B")
    bx.set_title("Normalized dose response", loc="left")
    for row in rows:
        color = MODEL_COLORS[row["label"]]
        ratio = row["half"] / row["mean"]
        bx.plot([0, 0.5, 1.0], [0, ratio, 1.0], marker="o" if not row["extension"] else "D", color=color, label=row["label"], zorder=3)
    bx.plot([0, 1], [0, 1], color=COLORS["neutral"], linestyle=":", linewidth=1.0, zorder=1)
    bx.set_xticks([0, 0.5, 1.0])
    bx.set_xlabel("Intervention strength $\\alpha$")
    bx.set_ylabel("Effect / full-dose effect")
    bx.set_ylim(-0.02, 1.04)
    clean_axis(bx, grid="y")

    panel_label(cx, "C")
    cx.set_title("Random-direction calibration", loc="left")
    jitter = np.linspace(-0.16, 0.16, 32)
    for yy, row in zip(y, rows):
        randoms = np.sort(np.asarray(row["randoms"], dtype=float))
        cx.scatter(randoms, yy + jitter, s=10, color=MODEL_COLORS[row["label"]], alpha=0.38, linewidth=0, zorder=2)
        cx.plot([-row["random_threshold"], row["random_threshold"]], [yy, yy], color=COLORS["neutral"], linewidth=2.8, alpha=0.35, solid_capstyle="round", zorder=1)
        cx.scatter(row["mean"], yy, marker="D", s=44, color=MODEL_COLORS[row["label"]], edgecolor="white", linewidth=0.6, zorder=4)
    cx.axvline(0, color=COLORS["ink"], linewidth=0.8)
    cx.set_yticks(y, [r["label"] for r in rows])
    cx.set_xlim(-0.09, 0.80)
    cx.set_xlabel("Mean high − low effect")
    clean_axis(cx, grid="x")
    cx.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=COLORS["neutral"], alpha=0.55, label="32 random directions"), Line2D([], [], marker="D", linestyle="none", color=COLORS["ink"], label="Transferred $C$")], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)

    save_figure(fig, "main_06_prospective_transfer", sources=[TRANSFER_PATH, EXTENSION_PATH], description="Prospective transfer, dose response, and random-direction calibration across the two-model replication and later checkpoint extension.", manifest=manifest)


def figure_selectivity(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    transfer_rows = data["transfer_rows"]
    transfer = data["transfer"]
    model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.12, 0.88]}, constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("Transferred $C$ is not a pure channel", loc="left")
    endpoint_defs = [("interaction", "Target interaction"), ("shared", "Shared-purpose collateral"), ("format", "Format collateral")]
    x = np.arange(len(endpoint_defs))
    offsets = [-0.14, 0.14]
    rng = np.random.default_rng(20260827)
    for offset, model_id in zip(offsets, model_ids):
        label = MODEL_LABELS[model_id]
        for idx, (endpoint, _) in enumerate(endpoint_defs):
            vals = np.asarray(list(endpoint_effects(transfer_rows, model_id, endpoint).values()))
            assert len(vals) == 80
            jitter = rng.normal(0, 0.018, size=len(vals))
            ax.scatter(np.full(len(vals), idx + offset) + jitter, vals, s=7, color=MODEL_COLORS[label], alpha=0.14, linewidth=0, zorder=1)
            low, high = bootstrap_ci(vals, seed=20260827 + idx)
            ax.errorbar(idx + offset, vals.mean(), yerr=np.array([[vals.mean() - low], [high - vals.mean()]]), fmt="o", color=MODEL_COLORS[label], markeredgecolor="white", markeredgewidth=0.55, capsize=2.2, zorder=4)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x, [label for _, label in endpoint_defs])
    ax.set_ylabel("High − low readout effect")
    ax.set_ylim(-0.07, 0.83)
    clean_axis(ax, grid="y")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=MODEL_COLORS[MODEL_LABELS[m]], label=MODEL_LABELS[m]) for m in model_ids], loc="upper right")

    panel_label(bx, "B")
    bx.set_title("Item-level effects do not align", loc="left")
    e8 = primary_effects(transfer_rows, model_ids[0])
    e70 = primary_effects(transfer_rows, model_ids[1])
    common = sorted(set(e8) & set(e70))
    xv = np.asarray([e8[k] for k in common])
    yv = np.asarray([e70[k] for k in common])
    bx.scatter(xv, yv, s=18, color=COLORS["purple"], alpha=0.62, edgecolor="white", linewidth=0.35)
    bx.axhline(yv.mean(), color=COLORS["muted"], linewidth=0.7, linestyle=":")
    bx.axvline(xv.mean(), color=COLORS["muted"], linewidth=0.7, linestyle=":")
    bx.set_xlabel("Llama-3.1-8B block effect")
    bx.set_ylabel("Llama-3.3-70B block effect")
    clean_axis(bx, grid="both")
    bx.text(0.04, 0.96, f"Pearson $r$={transfer['cross_model']['primary_effect_pearson_r']:.3f}\n$p$={transfer['cross_model']['primary_effect_pearson_p_two_sided']:.3f}", transform=bx.transAxes, ha="left", va="top", fontsize=7.2, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})

    save_figure(fig, "main_07_scope_and_selectivity", sources=[TRANSFER_PATH, TRANSFER_CSV], description="Target and collateral transfer responses plus the absence of cross-model block-level alignment.", manifest=manifest)


def figure_observational_depth(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    sweeps = data["sweeps"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), constrained_layout=True, sharex=True)
    definitions = [
        ("Held-out purpose", lambda r: r["decorrelated"]["auc"]),
        ("Cross-format minimum", lambda r: min(r["b2c"]["auc"], r["c2b"]["auc"])),
        ("Format", lambda r: r["format"]["auc"]),
    ]
    for idx, (ax, (title, accessor)) in enumerate(zip(axes, definitions)):
        panel_label(ax, chr(ord("A") + idx))
        ax.set_title(title, loc="left")
        for label in MODEL_ORDER:
            rows = sweeps[label]
            q = np.asarray([r["layer"] / (len(rows) - 1) for r in rows])
            vals = np.asarray([accessor(r) for r in rows])
            ax.plot(q, vals, color=MODEL_COLORS[label], label=label)
        ax.axhline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.45, 1.01)
        ax.set_xlabel("Normalized depth")
        clean_axis(ax, grid="y")
    axes[0].set_ylabel("ROC AUC")
    axes[-1].legend(loc="lower right")
    save_figure(fig, "supp_01_observational_depth", sources=list(data["sweep_paths"].values()), description="Full observational layer trajectories on a normalized-depth axis.", manifest=manifest)


def figure_estimator_complete(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    result = data["estimator"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for idx, (ax, metric, title) in enumerate(
        [
            (axes[0], "macro_format_roc_auc", "AUC contrast"),
            (axes[1], "macro_format_balanced_accuracy", "Balanced-accuracy contrast"),
        ]
    ):
        panel_label(ax, chr(ord("A") + idx))
        ax.set_title(title, loc="left")
        estimator_names = ["diagonal_lda", "logistic", "interaction"]
        y = np.arange(len(estimator_names))[::-1]
        for yy, name in zip(y, estimator_names):
            point = result["point_summary"]["aggregate"][name][metric]
            ci = result["bootstrap"]["aggregate"][name][metric]
            errorbarh(ax, yy, point, ci["ci_2.5"], ci["ci_97.5"], ESTIMATOR_COLORS[name], marker="s" if name == "diagonal_lda" else "o")
        ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
        ax.set_yticks(y, [ESTIMATOR_LABELS[n] for n in estimator_names])
        ax.xaxis.set_major_formatter(FuncFormatter(signed_formatter))
        ax.set_xlabel("Mixed − best specialist")
        clean_axis(ax, grid="x")
    save_figure(fig, "supp_02_estimator_endpoints", sources=[ESTIMATOR_PATH], description="Complete aggregate estimator comparison for both prespecified reporting endpoints.", manifest=manifest)


def figure_causal_components(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    causal = data["causal"]
    endpoint = {"A": "shared_purpose", "B": "format", "C": "purpose_interaction"}
    components = [("A", "$A$: shared purpose"), ("B", "$B$: format"), ("C", "$C$: interaction")]
    model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    high_layers = {model_ids[0]: 29, model_ids[1]: 74}
    low_layers = {model_ids[0]: 5, model_ids[1]: 13}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)
    for idx, (ax, (component, title)) in enumerate(zip(axes, components)):
        panel_label(ax, chr(ord("A") + idx))
        ax.set_title(title, loc="left")
        for model_idx, model_id in enumerate(model_ids):
            label = MODEL_LABELS[model_id]
            vals = []
            for layer in (low_layers[model_id], high_layers[model_id]):
                row = next(r for r in causal["response_metrics"] if r["model_id"] == model_id and r["source_layer"] == layer and r["component"] == component and r["endpoint"] == endpoint[component] and r["alpha"] == 1.0)
                vals.append(row)
            x = np.array([0, 1]) + (-0.06 if model_idx == 0 else 0.06)
            y = [r["point"] for r in vals]
            ax.plot(x, y, marker="o", color=MODEL_COLORS[label], label=label, zorder=3)
            for xx, row in zip(x, vals):
                ax.errorbar(xx, row["point"], yerr=np.array([[row["point"] - row["ci_2.5"]], [row["ci_97.5"] - row["point"]]]), fmt="none", ecolor=MODEL_COLORS[label], capsize=2.0, linewidth=1.0)
                symbol = "*" if row["random_p"] is not None and row["random_p"] <= 0.05 and row["ci_2.5"] > 0 else ""
                if symbol:
                    ax.text(xx, row["ci_97.5"] + 0.04 * max(0.02, max(abs(v) for v in y)), symbol, ha="center", va="bottom", fontsize=8, color=MODEL_COLORS[label])
        ax.axhline(0, color=COLORS["ink"], linewidth=0.75)
        ax.set_xticks([0, 1], ["Low-control", "High-interaction"])
        ax.tick_params(axis="x", labelrotation=12)
        clean_axis(ax, grid="y")
    axes[0].set_ylabel("Intended readout effect")
    axes[-1].legend(loc="upper left")
    axes[0].text(0.0, -0.32, "* CI excludes zero and random $p≤.05$", transform=axes[0].transAxes, fontsize=6.8, color=COLORS["muted"])
    save_figure(fig, "supp_03_causal_components", sources=[CAUSAL_PATH], description="All intended A, B, and C component responses at the frozen low-control and high-interaction layers.", manifest=manifest)


def figure_determinism_gate(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    invalid = data["invalid"]
    causal = data["causal"]
    before = {MODEL_LABELS[r["model_id"]]: r for r in invalid["failed_runs"]}
    labels = ["Llama-3.1-8B", "Llama-3.3-70B"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("Maximum absolute disagreement", loc="left")
    x = np.arange(2)
    for label in labels:
        ax.plot(x, [before[label]["observed_max_abs"], causal["reproduction"]["source_max_abs"]], marker="o", color=MODEL_COLORS[label], label=label, zorder=3)
    ax.axhline(0.125, color=COLORS["fail"], linestyle="--", linewidth=1.0)
    ax.text(0.98, 0.129, "frozen max", ha="right", va="bottom", fontsize=6.8, color=COLORS["fail"])
    ax.set_xticks(x, ["Invalid v2", "Deterministic v3"])
    ax.set_ylabel("max_abs")
    ax.set_ylim(-0.008, 0.275)
    clean_axis(ax, grid="y")
    ax.legend(loc="upper right")

    panel_label(bx, "B")
    bx.set_title("Cosine agreement", loc="left")
    for idx, label in enumerate(labels):
        bx.plot([0, 1], [before[label]["observed_cosine"], causal["reproduction"]["source_min_cosine"]], marker="o", color=MODEL_COLORS[label], label=label)
    bx.axhline(0.999, color=COLORS["fail"], linestyle="--", linewidth=1.0)
    bx.set_xticks([0, 1], ["Invalid v2", "Deterministic v3"])
    bx.set_ylim(0.9989, 1.00008)
    bx.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
    bx.set_ylabel("Minimum cosine")
    clean_axis(bx, grid="y")
    bx.legend(loc="upper left")
    bx.text(0.50, 0.145, "Cosine passed in both; max_abs was the binding gate", transform=bx.transAxes, ha="center", fontsize=6.9, color=COLORS["muted"])
    save_figure(fig, "supp_04_determinism_gate", sources=[INVALID_GATE_PATH, CAUSAL_PATH], description="Fail-closed source-activation reproduction gate before and after deterministic execution pinning.", manifest=manifest)


def figure_transfer_blocks(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    transfer_rows = data["transfer_rows"]
    extension_rows = data["extension_rows"]
    model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    arrays = [np.asarray(list(primary_effects(transfer_rows, m).values())) for m in model_ids]
    arrays.append(np.asarray(list(primary_effects(extension_rows).values())))
    labels = MODEL_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("All transferred payload-block effects", loc="left")
    positions = np.arange(3)
    parts = ax.violinplot(arrays, positions=positions, widths=0.64, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(MODEL_COLORS[label])
        body.set_edgecolor(MODEL_COLORS[label])
        body.set_alpha(0.20)
    rng = np.random.default_rng(20260827)
    for pos, vals, label in zip(positions, arrays, labels):
        ax.scatter(pos + rng.uniform(-0.13, 0.13, len(vals)), vals, s=8, color=MODEL_COLORS[label], alpha=0.40, linewidth=0)
        ax.scatter(pos, vals.mean(), marker="D", s=38, color=MODEL_COLORS[label], edgecolor="white", linewidth=0.55, zorder=4)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=12)
    ax.set_ylabel("High − low interaction effect")
    clean_axis(ax, grid="y")

    panel_label(bx, "B")
    bx.set_title("Empirical cumulative distributions", loc="left")
    for vals, label in zip(arrays, labels):
        ordered = np.sort(vals)
        p = np.arange(1, len(ordered) + 1) / len(ordered)
        bx.step(ordered, p, where="post", color=MODEL_COLORS[label], label=label)
    bx.axvline(0, color=COLORS["ink"], linewidth=0.8)
    bx.set_xlabel("High − low interaction effect")
    bx.set_ylabel("Cumulative fraction")
    bx.set_ylim(0, 1.02)
    clean_axis(bx, grid="y")
    bx.legend(loc="lower right")
    save_figure(fig, "supp_05_transfer_block_distributions", sources=[TRANSFER_CSV, EXTENSION_CSV], description="Distributions and empirical CDFs of all 80 block-level transfer effects in each checkpoint.", manifest=manifest)


def figure_extension_topics(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    robustness = data["extension"]["topic_robustness_descriptive"]
    topics = sorted(robustness["topic_means"])
    means = np.asarray([robustness["topic_means"][t] for t in topics])
    loo = np.asarray([robustness["leave_one_topic_out_means"][t] for t in topics])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    ax, bx = axes
    panel_label(ax, "A")
    ax.set_title("Extension effect by topic", loc="left")
    y = np.arange(len(topics))[::-1]
    ax.scatter(means, y, color=MODEL_COLORS["Llama-3.1-70B"], s=24, edgecolor="white", linewidth=0.4)
    ax.axvline(data["extension"]["primary_high_minus_low"]["mean"], color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax.set_yticks(y, topics)
    ax.set_xlabel("Topic mean effect")
    clean_axis(ax, grid="x")

    panel_label(bx, "B")
    bx.set_title("Leave-one-topic-out stability", loc="left")
    bx.scatter(loo, y, color=MODEL_COLORS["Llama-3.1-70B"], s=24, edgecolor="white", linewidth=0.4)
    bx.axvline(data["extension"]["primary_high_minus_low"]["mean"], color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    bx.set_yticks(y, topics)
    bx.set_xlabel("Mean after omitting topic")
    clean_axis(bx, grid="x")
    bx.text(0.98, 0.02, "Descriptive robustness; not a new decision family", transform=bx.transAxes, ha="right", va="bottom", fontsize=6.8, color=COLORS["muted"])
    save_figure(fig, "supp_06_extension_topic_robustness", sources=[EXTENSION_PATH], description="Descriptive topic and leave-one-topic-out robustness for the prospective Llama-3.1-70B extension.", manifest=manifest)


def figure_equal_n(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    eq = data["equal_n"]
    keys = ["benchmark_only", "casual_only", "equal_n_mixed"]
    labels = ["Benchmark-only", "Casual-only", "Equal-N mixed"]
    values = [[float(row[k]["auc"]) for row in eq["results"]] for k in keys]
    fig, ax = plt.subplots(figsize=(3.35, 2.65), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("8B equal-training-size control", loc="left")
    x = np.arange(3)
    for seed_idx in range(len(eq["results"])):
        ax.plot(x, [values[k][seed_idx] for k in range(3)], color=COLORS["neutral"], alpha=0.28, linewidth=0.8, zorder=1)
    for idx, vals in enumerate(values):
        ax.scatter(np.full(len(vals), idx), vals, s=22, color=COLORS["blue"] if idx < 2 else COLORS["green"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(idx, np.mean(vals), marker="D", s=38, color=COLORS["blue"] if idx < 2 else COLORS["green"], edgecolor="white", linewidth=0.55, zorder=4)
    ax.set_xticks(x, labels, rotation=14)
    ax.set_ylabel("Held-out purpose AUC")
    ax.set_ylim(0.945, 0.985)
    clean_axis(ax, grid="y")
    ax.text(0.98, 0.03, "4 fixed seeds · layer 12 · development-only", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"])
    save_figure(fig, "supp_07_equal_n_8b", sources=[EQUAL_N_PATH], description="Development-only equal-training-size comparison for Llama-3.1-8B at layer 12.", manifest=manifest)


def figure_jlens(data: dict[str, Any], manifest: dict[str, Any]) -> None:
    assay = data["jlens"]
    layers = assay["source_layers"]
    purpose = [assay["summary"]["purpose"][str(layer)]["oriented_auc"] for layer in layers]
    fmt = [assay["summary"]["format"][str(layer)]["oriented_auc"] for layer in layers]
    fig, ax = plt.subplots(figsize=(3.35, 2.65), constrained_layout=True)
    panel_label(ax, "A")
    ax.set_title("8B Jacobian-lens diagnostic", loc="left")
    ax.plot(layers, purpose, marker="o", color=COLORS["blue"], label="Stated purpose")
    ax.plot(layers, fmt, marker="s", color=COLORS["vermillion"], label="No-cue format")
    ax.axhline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(layers)
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Oriented ROC AUC")
    ax.set_ylim(0.48, 0.70)
    clean_axis(ax, grid="y")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.03, "Direct scalar readout · development-only", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"])
    save_figure(fig, "supp_08_jlens_diagnostic", sources=[JLENS_PATH], description="Development-only direct Jacobian-lens purpose and no-cue format diagnostic.", manifest=manifest)


def make_contact_sheet(stems: list[str], name: str, title: str) -> Path:
    images = [plt.imread(OUT / f"{stem}.png") for stem in stems]
    ncols = 2
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.2 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, image, stem in zip(axes_array, images, stems):
        ax.imshow(image)
        ax.set_title(stem, loc="left", fontsize=11, fontweight="bold")
        ax.axis("off")
    for ax in axes_array[len(images):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def load_all() -> dict[str, Any]:
    sweep_paths = {
        "Llama-3.1-8B": PINNED_DIR / "layer_sweep_full320-v3_llama31_8b.json",
        "Llama-3.1-70B": PINNED_DIR / "layer_sweep_full320-v3_llama31_70b.json",
        "Llama-3.3-70B": PINNED_DIR / "layer_sweep_full320-v3_llama33_70b.json",
    }
    return {
        "estimator": load_json(ESTIMATOR_PATH),
        "causal": load_json(CAUSAL_PATH),
        "transfer": load_json(TRANSFER_PATH),
        "transfer_rows": load_csv(TRANSFER_CSV),
        "extension": load_json(EXTENSION_PATH),
        "extension_rows": load_csv(EXTENSION_CSV),
        "invalid": load_json(INVALID_GATE_PATH),
        "equal_n": load_json(EQUAL_N_PATH),
        "jlens": load_json(JLENS_PATH),
        "sweeps": {label: load_json(path) for label, path in sweep_paths.items()},
        "sweep_paths": sweep_paths,
    }


def main() -> None:
    configure_style()
    data = load_all()
    validation = validate_contracts(data)
    manifest: dict[str, Any] = {}

    figure_design_overview(data, manifest)
    figure_cross_format_gap(data, manifest)
    figure_format_gap_regrowth(data, manifest)
    figure_estimator_reversal(data, manifest)
    figure_causal_decision(data, manifest)
    figure_causal_linearity(data, manifest)
    figure_prospective_transfer(data, manifest)
    figure_selectivity(data, manifest)

    figure_observational_depth(data, manifest)
    figure_estimator_complete(data, manifest)
    figure_causal_components(data, manifest)
    figure_determinism_gate(data, manifest)
    figure_transfer_blocks(data, manifest)
    figure_extension_topics(data, manifest)
    figure_equal_n(data, manifest)
    figure_jlens(data, manifest)

    main_stems = sorted(stem for stem in manifest if stem.startswith("main_"))
    supp_stems = sorted(stem for stem in manifest if stem.startswith("supp_"))
    contact_main = make_contact_sheet(main_stems, "contact_sheet_main.png", "Main-paper figure candidates")
    contact_supp = make_contact_sheet(supp_stems, "contact_sheet_supplement.png", "Supplementary figure candidates")

    output = {
        "schema_version": 1,
        "status": "complete_validated_publication_figure_suite",
        "style_contract": {
            "width_inches": {"single_column": 3.35, "double_column": 7.0},
            "font": "Arial/Helvetica/DejaVu Sans fallback",
            "model_colors": MODEL_COLORS,
            "formats": ["pdf", "svg", "png"],
            "png_dpi": 400,
            "notes": "Colorblind-safe palette; model colors and estimator encodings are fixed across panels.",
        },
        "validation": validation,
        "figures": manifest,
        "contact_sheets": {
            "main": {"path": rel(contact_main), "sha256": sha256_file(contact_main)},
            "supplement": {"path": rel(contact_supp), "sha256": sha256_file(contact_supp)},
        },
    }
    manifest_path = OUT / "figure_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "n_figures": len(manifest), "output_dir": rel(OUT), "manifest_sha256": sha256_file(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
