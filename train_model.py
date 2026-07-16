"""
Step 3. Train the classifier.

Usage:
    python train_model.py

THE THING THAT MATTERS IN HERE
------------------------------
Frames recorded inside one 5-second burst are near-duplicates of each other.
A plain random train/test split scatters those duplicates across both sides,
so the model gets tested on frames it effectively already saw. You get 99%
and it means nothing.

So this script reports two numbers:

  naive split  - random shuffle, leaks, optimistic, what every tutorial prints
  group split  - whole bursts held out, no leakage, what's actually true

The gap between them is the single most interesting thing this project
produces. Report the group number publicly. Report both to be interesting.

EVALUATION ARTIFACTS
---------------------
Everything lands in outputs/, styled to match the app's own dark palette so
a screenshot of the demo and a screenshot of these charts read as one post:

  confusion_matrix.png     which letters get mistaken for which
  per_letter_accuracy.png  recall per letter, green/amber/red at 90%/70%
  per_letter_f1.png        same idea, F1 (precision+recall balance)
  leakage_comparison.png   naive vs honest, side by side - the one chart
                           that explains why the honest number is lower
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

# ---------------------------------------------------------------- chart style
# Same Apple-system palette as the live app (config.py's COL_* constants,
# converted from BGR to hex) so these plots and the on-screen UI read as one
# design, not a mismatched matplotlib default dropped next to it. Status
# colors (good/warn/critical) are validated for contrast against SURFACE -
# see the dataviz skill's color-formula.md - and kept distinct from the
# categorical pair used for the naive-vs-honest comparison.
SURFACE = "#1C1C1E"
PANEL = "#2C2C2E"
TEXT = "#F5F5F7"
TEXT_DIM = "#98989D"
TEXT_FAINT = "#636366"
GRID = "#38383A"
GOOD = "#30D158"     # >= 90%
WARN = "#FF9F0A"     # 70-90%
CRIT = "#FF453A"     # < 70%
NAIVE_COL = "#FF453A"   # leaky/optimistic - the one to distrust
HONEST_COL = "#0A84FF"  # group-split - the real number


def _status_color(frac: float) -> str:
    return GOOD if frac >= 0.90 else WARN if frac >= 0.70 else CRIT


def _dark_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT)
    return fig, ax


def build_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=600,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            ),
        ),
    }


def load():
    if not config.DATASET_CSV.exists():
        sys.exit(f"no dataset at {config.DATASET_CSV} - run collect_data.py first")
    df = pd.read_csv(config.DATASET_CSV)
    if "group" not in df.columns:
        df["group"] = df["label"] + "_0"
    feat_cols = [c for c in df.columns if c not in ("label", "group")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    # sklearn's MLP + early_stopping chokes on string targets (it np.isnan's the
    # predictions), so integer-encode. classes_ stays alphabetical, which keeps
    # predict_proba column order == labels order for inference.
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    g = df["group"].to_numpy()
    return X, y, g, df, list(le.classes_)


def class_report(df):
    counts = df["label"].value_counts()
    groups = df.groupby("label")["group"].nunique()
    print(f"\n{len(df)} samples, {df['label'].nunique()} classes\n")
    thin = [L for L in config.LETTERS if counts.get(L, 0) < config.SAMPLES_PER_LETTER * 0.6]
    if thin:
        print(f"!! underrepresented, go record more: {thin}")
    single = [L for L in config.LETTERS if groups.get(L, 0) < 2]
    if single:
        print(f"!! only 1 burst for {single} -> group split will be unreliable for them")
    missing = [L for L in config.LETTERS if counts.get(L, 0) == 0]
    if missing:
        print(f"!! MISSING ENTIRELY: {missing}")
    return len(single) == 0


def evaluate(X, y, split, tag):
    tr, te = split
    rows = {}
    for name, model in build_models().items():
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        acc = accuracy_score(y[te], pred)
        rows[name] = (acc, model, pred, te)
        print(f"  {tag:12s} {name:14s} {acc*100:6.2f}%")
    return rows


def leave_one_burst_out(y, g, seed=42):
    """
    Hold out ONE whole recording burst per letter.

    GroupShuffleSplit is wrong here: it picks groups globally at random, so it
    happily hands you a test set that's missing 9 of your 24 letters. This
    guarantees every class is represented in test, and that no frame in test
    came from a burst that's also in train.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    test, skipped = [], []
    for cls in np.unique(y):
        m = y == cls
        bursts = np.unique(g[m])
        if len(bursts) < 2:
            skipped.append(cls)
            continue
        held = rng.choice(bursts)
        test.append(idx[m & (g == held)])
    if not test:
        return None, None, skipped
    te = np.concatenate(test)
    tr = np.setdiff1d(idx, te)
    return tr, te, skipped


# ---------------------------------------------------------------- plots
def plot_confusion_matrix(cm, labels, acc, best_name, label, path):
    fig, ax = _dark_axes((9, 8))
    im = ax.imshow(cm, cmap="magma")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"{best_name}  {acc*100:.1f}%  [{label}]", fontsize=12, pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color=TEXT if i != j else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelcolor=TEXT_DIM, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_status_bar(labels, values, path, title, subtitle, value_fmt="{:.0%}"):
    """
    Horizontal per-letter bar, sorted worst-to-best, colored by status
    (green >= 90%, amber 70-90%, red < 70%) with a labelled 90% line so the
    "is it above 90%" question the numbers answer is visible at a glance,
    not something you have to read off an axis.
    """
    order = np.argsort(values)
    labels_sorted = [labels[i] for i in order]
    values_sorted = [values[i] for i in order]
    colors = [_status_color(v) for v in values_sorted]

    fig, ax = _dark_axes((7, 9))
    y_pos = np.arange(len(labels_sorted))
    ax.barh(y_pos, values_sorted, color=colors, height=0.68, zorder=3)
    for yi, v in zip(y_pos, values_sorted):
        ax.text(min(v + 0.015, 0.965), yi, value_fmt.format(v), va="center",
                fontsize=8, color=TEXT)

    ax.axvline(0.90, color=TEXT_FAINT, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.90, len(labels_sorted) - 0.2, " 90%", fontsize=8, color=TEXT_DIM,
            ha="left", va="bottom")

    ax.set_yticks(y_pos, labels_sorted, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(subtitle, fontsize=9, color=TEXT_DIM, pad=12, loc="left")
    fig.suptitle(title, fontsize=13, color=TEXT, x=0.125, y=0.975, ha="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, WARN, CRIT)]
    leg = ax.legend(handles, ["≥ 90%", "70-90%", "< 70%"], loc="lower right",
                     frameon=False, fontsize=8, ncol=1)
    for t in leg.get_texts():
        t.set_color(TEXT_DIM)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_leakage_comparison(naive_by_model, honest_by_model, path):
    """
    The single most shareable chart this project produces: naive (leaky,
    optimistic) vs honest (group-split) accuracy, side by side, per model.
    The gap between the two bars in each pair *is* the data-leakage story.
    """
    models = list(naive_by_model.keys())
    naive_vals = [naive_by_model[m] for m in models]
    honest_vals = [honest_by_model.get(m) for m in models]

    fig, ax = _dark_axes((7, 5))
    x = np.arange(len(models))
    w = 0.32
    b1 = ax.bar(x - w / 2, naive_vals, width=w, color=NAIVE_COL, zorder=3, label="naive split (leaky)")
    have_honest = [v is not None for v in honest_vals]
    b2 = ax.bar(x[have_honest] + w / 2, [v for v in honest_vals if v is not None],
                width=w, color=HONEST_COL, zorder=3, label="group split (honest)")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h*100:.1f}%",
                    ha="center", fontsize=9, color=TEXT)

    if all(have_honest):
        for xi, (nv, hv) in enumerate(zip(naive_vals, honest_vals)):
            gap = (nv - hv) * 100
            ax.text(xi, max(nv, hv) + 0.07, f"+{gap:.1f} pts inflation",
                    ha="center", fontsize=8, color=TEXT_FAINT, style="italic")

    ax.set_xticks(x, [m.replace("_", " ") for m in models])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Naive vs honest accuracy - the leakage gap", fontsize=13, pad=12, loc="left")

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                     frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT_DIM)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    X, y, g, df, labels = load()
    class_report(df)
    print(f"\nfeatures: {X.shape[1]}   rotation_norm: {config.USE_ROTATION_NORM}\n")

    # ---- naive split (leaky, for comparison only)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    naive = evaluate(X, y, (tr, te), "naive")

    # ---- group split (honest)
    honest = None
    tr_g, te_g, skipped = leave_one_burst_out(y, g)
    if tr_g is None:
        print("  no letter has 2+ bursts -> cannot measure honestly. Go record more.")
    else:
        if skipped:
            print(f"  (only 1 burst, excluded from honest test: "
                  f"{[labels[i] for i in skipped]})")
        honest = evaluate(X, y, (tr_g, te_g), "group")

    source = honest if honest else naive
    label = "group-split (honest)" if honest else "naive-split (OPTIMISTIC - leaky)"
    best_name = max(source, key=lambda k: source[k][0])
    acc, model, pred, te = source[best_name]

    print(f"\nwinner: {best_name} @ {acc*100:.2f}%  [{label}]")
    if honest:
        gap = (naive[best_name][0] - honest[best_name][0]) * 100
        print(f"leakage inflation: naive is {gap:+.2f} pts higher than reality")

    ids = list(range(len(labels)))
    print("\n" + classification_report(y[te], pred, labels=ids, target_names=labels,
                                       zero_division=0))

    # ---- per-letter accuracy + F1, printed plainly with the 90% bar user asked for
    cm = confusion_matrix(y[te], pred, labels=ids)
    per_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    _, _, per_f1, _ = precision_recall_fscore_support(y[te], pred, labels=ids, zero_division=0)

    print(f"per-letter accuracy [{label}]:")
    order = np.argsort(per_acc)[::-1]
    n_at_90 = 0
    for i in order:
        flag = "OK  >=90%" if per_acc[i] >= 0.90 else ("warn 70-90%" if per_acc[i] >= 0.70 else "LOW <70%")
        n_at_90 += per_acc[i] >= 0.90
        print(f"  {labels[i]:2s}  acc {per_acc[i]*100:5.1f}%   f1 {per_f1[i]*100:5.1f}%   {flag}")
    print(f"\n{n_at_90}/{len(labels)} letters at or above 90% accuracy "
          f"[{label}]")

    print("\nweakest letters:")
    for i in order[-1:-7:-1]:
        wrong = cm[i].copy(); wrong[i] = 0
        conf = labels[int(np.argmax(wrong))] if wrong.sum() else "-"
        print(f"  {labels[i]}  {per_acc[i]*100:5.1f}%   mostly mistaken for {conf}")

    # ---- charts
    cm_png = config.OUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, labels, acc, best_name, label, cm_png)
    print(f"\nsaved {cm_png}")

    acc_png = config.OUT_DIR / "per_letter_accuracy.png"
    plot_status_bar(labels, per_acc, acc_png, "Per-letter accuracy",
                     f"{best_name}, {label}  ·  {n_at_90}/{len(labels)} letters ≥ 90%")
    print(f"saved {acc_png}")

    f1_png = config.OUT_DIR / "per_letter_f1.png"
    plot_status_bar(labels, per_f1, f1_png, "Per-letter F1 score",
                     f"{best_name}, {label}  ·  precision + recall balance", value_fmt="{:.2f}")
    print(f"saved {f1_png}")

    if honest:
        naive_by_model = {m: naive[m][0] for m in naive}
        honest_by_model = {m: honest[m][0] for m in honest}
        leak_png = config.OUT_DIR / "leakage_comparison.png"
        plot_leakage_comparison(naive_by_model, honest_by_model, leak_png)
        print(f"saved {leak_png}")

    joblib.dump(
        {"model": model, "labels": labels, "name": best_name,
         "accuracy": float(acc), "eval": label,
         "rotation_norm": config.USE_ROTATION_NORM},
        config.CLASSIFIER_PKL,
    )
    print(f"saved {config.CLASSIFIER_PKL}")


if __name__ == "__main__":
    main()
