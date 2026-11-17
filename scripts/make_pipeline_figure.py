"""Draw reports/figures/pipeline_overview.png: the study's pipeline, for the revised paper.

Usage: uv run python scripts/make_pipeline_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from blindspy.data import ROOT  # noqa: E402

OUT = ROOT / "reports" / "figures" / "pipeline_overview.png"

STAGES = [
    ("ISCXVPN2016\n16 ARFF task files\n(4 timeouts)", "#dbe9f6"),
    ("Column audit\nvalue identities\n(9 files renamed)", "#dbe9f6"),
    ("Cleaning\nexact duplicates out\nbefore splitting", "#dbe9f6"),
    ("Stratified split\n70 / 15 / 15\nseed 42", "#fde9d4"),
    ("Model search\n7 method types\n5-fold CV, Optuna", "#fde9d4"),
    ("Selection\nvalidation macro F1\n(test sealed)", "#fde9d4"),
    ("Test, once\n10,000 paired\nbootstrap, Holm", "#e2f0d9"),
    ("Calibration\n+ SHAP, demo,\nCLI predictor", "#e2f0d9"),
]


def main():
    fig, ax = plt.subplots(figsize=(10, 2.6), dpi=200)
    ax.set_xlim(0, len(STAGES))
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    w, h, y = 0.86, 0.62, 0.22
    for i, (label, color) in enumerate(STAGES):
        x = i + (1 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, ec="#333333", lw=0.8))  # fmt: skip
        ax.text(i + 0.5, y + h / 2, label, ha="center", va="center", fontsize=7)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (i + 1 + (1 - w) / 2, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=8, lw=0.8))  # fmt: skip
    for (a, b, text) in [(0, 3, "Data preparation (no model involved)"),
                         (3, 6, "Development on training + validation rows"),
                         (6, 8, "Final evaluation and use")]:  # fmt: skip
        ax.annotate("", xy=(a + 0.07, 1.0), xytext=(b - 0.07, 1.0),
                    arrowprops=dict(arrowstyle="|-|", lw=0.7, color="#555555"))  # fmt: skip
        ax.text((a + b) / 2, 1.08, text, ha="center", va="bottom", fontsize=7.5, color="#222222")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
