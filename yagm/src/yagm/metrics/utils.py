import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle


def plot_confusion_matrix(
    cm, class_names, number_fmt="f", wrong_top_k=None, normalize_rows=True
):
    """
    Render a confusion matrix to an RGB numpy array.
    - Green border on diagonal (drawn last).
    - Red border on off-diagonals:
        * all if wrong_top_k=None
        * else top-K per row (actual).
    - Labels on all four sides (top/right duplicated).
    - Colorbar at far right.
    - If normalize_rows=True, the colormap is scaled per row (each row shows relative distribution).
    """
    cm = np.asarray(cm)
    n_classes = len(class_names)
    display_names = [f"{i} - {name}" for i, name in enumerate(class_names)]
    # ---- Row normalization for visualization ----
    if normalize_rows:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_display = cm / row_sums
    else:
        cm_display = cm
    # ---- Layout constants ----
    AX_LEFT, AX_BOTTOM, AX_W, AX_H = 0.16, 0.18, 0.60, 0.62
    CBAR_LEFT, CBAR_BOTTOM, CBAR_W, CBAR_H = 0.93, 0.18, 0.02, 0.62
    RIGHT_TICK_PAD = 3
    TOP_TICK_PAD = 8
    fig_w = max(12, 1.2 * n_classes)
    fig_h = max(12, 1.2 * n_classes)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_W, AX_H])
    cax = fig.add_axes([CBAR_LEFT, CBAR_BOTTOM, CBAR_W, CBAR_H])
    # --- Heatmap ---
    sns.heatmap(
        cm_display,
        annot=False,  # we'll add text manually
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar=True,
        cbar_ax=cax,
        vmin=0.0,
        vmax=1.0 if normalize_rows else None,
    )
    # --- Annotate with actual counts (not normalized) ---
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j + 0.5,
                i + 0.5,
                format(cm[i, j], number_fmt),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    # --- Select off-diagonals to highlight (red) ---
    highlight_pairs = set()
    if wrong_top_k is None:
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    highlight_pairs.add((i, j))
    else:
        k = int(wrong_top_k)
        if k > 0:
            for i in range(n_classes):
                row_vals = [
                    (cm[i, j], j) for j in range(n_classes) if j != i and cm[i, j] > 0
                ]
                row_vals.sort(key=lambda x: x[0], reverse=True)
                for _, j in row_vals[: min(k, len(row_vals))]:
                    highlight_pairs.add((i, j))
    # Borders: red first
    for i, j in highlight_pairs:
        ax.add_patch(
            Rectangle((j, i), 1, 1, fill=False, edgecolor="red", linewidth=1.8)
        )
    # Then green diag
    for i in range(n_classes):
        ax.add_patch(
            Rectangle((i, i), 1, 1, fill=False, edgecolor="green", linewidth=2.6)
        )
    # ---- Duplicate labels (top/right) ----
    centers = np.arange(n_classes) + 0.5
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels(display_names, rotation=45, ha="left", va="bottom")
    ax_top.tick_params(axis="x", pad=TOP_TICK_PAD, length=3)
    ax_top.patch.set_alpha(0.0)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(centers)
    ax_right.set_yticklabels(display_names, rotation=0, va="center")
    ax_right.tick_params(axis="y", pad=RIGHT_TICK_PAD, length=3)
    ax_right.patch.set_alpha(0.0)
    for k_sp, spine in ax_right.spines.items():
        if k_sp != "right":
            spine.set_visible(False)
    # Base labels/title
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top")
    plt.setp(ax.get_yticklabels(), rotation=0)
    # ---- Rasterize to numpy ----
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(
        canvas.get_width_height()[::-1] + (4,)
    )
    plt.close(fig)
    return img[:, :, :3]
