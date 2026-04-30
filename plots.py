import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


MODELS = [
    "Naive\nModel",
    "Logistic\nRegression",
    "Random\nForest",
    "XGBoost",
    "Naive\nBayes",
    "RNN",
]

METRICS = {
    "Accuracy": [0.5477, 0.7128, 0.7592, 0.7709, 0.7340, 0.7697],
    "Precision": [0.3362, 0.5495, 0.6140, 0.5873, 0.5547, 0.6468],
    "Recall": [0.3383, 0.5841, 0.6212, 0.6256, 0.5795, 0.6104],
    "F1": [0.3110, 0.5542, 0.6064, 0.6017, 0.5610, 0.6094],
}

METRIC_COLORS = {
    "Accuracy": "#3C5488",
    "Precision": "#E64B35",
    "Recall": "#00A087",
    "F1": "#4DBBD5",
}

CLASS_LABELS = ["On-time", "Moderate", "Severe"]

CMS = {
    "Naive Model": np.array(
        [
            [662_763, 416_129, 1_026],
            [138_898, 92_808, 207],
            [41_535, 26_318, 64],
        ]
    ),
    "Logistic Regression": np.array(
        [
            [820_530, 243_285, 16_103],
            [73_098, 135_088, 23_727],
            [20_336, 19_728, 27_853],
        ]
    ),
    "Random Forest": np.array(
        [
            [871_905, 195_128, 12_885],
            [72_608, 146_875, 12_430],
            [19_760, 19_442, 28_715],
        ]
    ),
    "XGBoost": np.array(
        [
            [893_981, 159_952, 25_985],
            [76_545, 139_173, 16_195],
            [20_727, 16_708, 30_482],
        ]
    ),
    "Naive Bayes": np.array(
        [
            [862_248, 201_567, 16_103],
            [86_518, 122_452, 22_943],
            [23_562, 16_376, 27_979],
        ]
    ),
    "RNN": np.array(
        [
            [888_122, 184_345, 7_451],
            [75_009, 149_035, 7_869],
            [21_086, 21_970, 24_861],
        ]
    ),
}


def make_bar_chart():
    n_models = len(MODELS)
    n_metrics = len(METRICS)
    x = np.arange(n_models)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for i, (metric, values) in enumerate(METRICS.items()):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=METRIC_COLORS[metric],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, values):
            if v <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Score (macro avg)")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Model Comparison — 3-Class Delay Prediction", pad=12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "bar_chart.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_cm(ax, cm, title):
    if cm.sum() == 0:
        ax.text(
            0.5,
            0.5,
            f"{title}\n(fill in CMS)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#999999",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    normalized = cm / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "custom_blues",
        ["#f5f5f5", "#f5f5f5", "#9ecae1", "#2171b5", "#08306B"],
        N=256,
    )

    ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    n = len(CLASS_LABELS)
    for i in range(n):
        for j in range(n):
            value = cm[i, j]
            text_color = "white" if normalized[i, j] > 0.55 else "#171717"
            label = f"{value / 1000:.0f}k" if value >= 1000 else f"{value}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.set_yticklabels(CLASS_LABELS, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(0.6)


def make_confusion_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    order = [
        "Naive Model",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Naive Bayes",
        "RNN",
    ]

    for ax, name in zip(axes.flat, order):
        _plot_cm(ax, CMS[name], name)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


ROC_DATA = "roc_data.npz"

ROC_COLORS = {
    "Naive Model":         "#999999",
    "Logistic Regression": "#3C5488",
    "Random Forest":       "#00A087",
    "XGBoost":             "#E64B35",
    "Naive Bayes":         "#8172B2",
    "RNN":                 "#4DBBD5",
}


def make_roc_curves():
    """
    Multi-line ROC plot (one curve per model, macro-averaged across classes).
    Requires roc_data.npz with keys: y_test, <ModelName>_proba (one per model).
    Generate it with the notebook snippet in README of this script.
    """
    if not os.path.exists(ROC_DATA):
        print(f"Skipping ROC: {ROC_DATA} not found. "
              f"See snippet at the bottom of plots.py to generate it.")
        return

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    data = np.load(ROC_DATA)
    y_test = data["y_test"]
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#cccccc", linewidth=1, label="Chance (AUC = 0.50)")

    model_aucs = []
    for name in ["Naive Model", "Logistic Regression", "Random Forest",
                 "XGBoost", "Naive Bayes", "RNN"]:
        key = f"{name}_proba"
        if key not in data.files:
            continue
        y_proba = data[key]

        # Macro-average ROC across classes
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, linewidth=2,
                color=ROC_COLORS.get(name, "#444444"),
                label=f"{name} (AUC = {roc_auc:.3f})")
        model_aucs.append((name, roc_auc))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 3-Class Delay Prediction (macro-avg)", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_bar_chart()
    make_confusion_matrices()
    make_roc_curves()
    print(f"Done. Figures saved to ./{OUT_DIR}/")


# ---------------------------------------------------------------------------
# To generate roc_data.npz, paste this in a notebook cell after all 5 models
# are trained (LR, RF, XGB, NB, RNN) and the naive baseline exists:
#
#   import numpy as np
#   probas = {
#       "Naive Model_proba":         naive_model.predict_proba(X_test_preprocessed),
#       "Logistic Regression_proba": lr_model.predict_proba(X_test_preprocessed),
#       "Random Forest_proba":       rf_model.predict_proba(X_test_preprocessed),
#       "XGBoost_proba":             xgb_model.predict_proba(X_test_preprocessed),
#       "Naive Bayes_proba":         nb_clf.predict_proba(X_test_nb),
#       "RNN_proba":                 rnn_model.predict(X_test_rnn, verbose=0),
#   }
#   np.savez("roc_data.npz", y_test=y_test.values, **probas)
# ---------------------------------------------------------------------------
