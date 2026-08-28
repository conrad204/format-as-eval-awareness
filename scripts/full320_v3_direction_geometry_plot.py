import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/full320_v3_direction_geometry.json"))
ranks = d["ranks"]
models = ["llama31_8b", "llama31_70b", "llama33_70b"]
depths = ["spike", "min_gap", "final"]
colors = {"llama31_8b": "tab:blue", "llama31_70b": "tab:orange", "llama33_70b": "tab:green"}
markers = {"spike": "o", "min_gap": "s", "final": "^"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for model in models:
    for depth in depths:
        bd = d["by_model"][model]["by_depth"][depth]["by_rank"]
        e_purpose = [bd[str(r)]["E_purpose_in_format_subspace"] for r in ranks]
        frac_top64 = [bd[str(r)]["frac_format_subspace_in_top64_pcs"] for r in ranks]
        axes[0].plot(ranks, e_purpose, marker=markers[depth], color=colors[model], alpha=0.85,
                     linewidth=1.3, markersize=5)
        axes[1].plot(ranks, frac_top64, marker=markers[depth], color=colors[model], alpha=0.85,
                     linewidth=1.3, markersize=5)

axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ranks); axes[0].set_xticklabels(ranks)
axes[0].set_xlabel("matched-format PCA rank r")
axes[0].set_ylabel(r"$E_{purpose \to format}(r) = \|Q_{format}(r)^T w_{purpose}\|^2$")
axes[0].set_title("Purpose energy inside matched-format subspace")
axes[0].set_ylim(-0.001, 0.02)
axes[0].axhline(0, color="black", linewidth=0.5)

axes[1].set_xscale("log", base=2)
axes[1].set_xticks(ranks); axes[1].set_xticklabels(ranks)
axes[1].set_xlabel("matched-format PCA rank r")
axes[1].set_ylabel("fraction of matched-format subspace\ninside top-64 activation PCs")
axes[1].set_title("Matched-format subspace vs. high-variance subspace")
axes[1].set_ylim(0.5, 1.02)

model_handles = [plt.Line2D([0], [0], color=colors[m], lw=2, label=m) for m in models]
depth_handles = [plt.Line2D([0], [0], color="gray", marker=markers[dep], linestyle="",
                             label=dep) for dep in depths]
fig.legend(handles=model_handles + depth_handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.08), fontsize=9)
fig.suptitle("Matched-format subspace geometry: 3 checkpoints x 3 depths x rank 1/4/16/64", y=1.02)
fig.tight_layout()
out_png = "results/full320_v3_direction_geometry_figure.png"
fig.savefig(out_png, dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", out_png)
