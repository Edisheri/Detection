"""
Профессиональная визуализация метрик модели LungDx Pro.

Создаёт:
  reports/diagrams/metrics_full_report.png   — сводный лист (матрицы + бары + динамика + датасет)
  reports/diagrams/metrics_interpretation.png — интерпретационная карта

Источники данных:
  reports/metrics_summary.json   — метрики оценки (evaluate.py)
  weights/training_history.json  — динамика обучения (train.py)

Запуск: .venv\\Scripts\\python.exe scripts/generate_metrics_visual.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.ticker as mticker
import seaborn as sns

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = ROOT / "reports" / "metrics_summary.json"
HISTORY_PATH = ROOT / "weights"  / "training_history.json"
IMG_EXTS_SCAN = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
CLASS_ORDER = ["Cancer", "COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

# ── Тёмная профессиональная тема ─────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#21262D"
BLUE    = "#1F6FEB"
BLUE2   = "#58A6FF"
GREEN   = "#3FB950"
RED     = "#F85149"
ORANGE  = "#E3B341"
PURPLE  = "#8957E5"
TEXT    = "#E6EDF3"
SUBTEXT = "#8B949E"
GRID    = "#21262D"

CLASS_COLORS = {
    "COVID-19":     BLUE2,
    "Cancer":       RED,
    "Normal":       GREEN,
    "Pneumonia":    ORANGE,
    "Tuberculosis": PURPLE,
}
CLASSES_RU = {
    "COVID-19":     "COVID-19",
    "Cancer":       "Рак лёгкого",
    "Normal":       "Норма",
    "Pneumonia":    "Пневмония",
    "Tuberculosis": "Туберкулёз",
}

# Fallback, если недоступен импорт config или папки пусты
DATASET_FALLBACK = {
    "train": {
        "Cancer":       5546,
        "COVID-19":     2882,
        "Normal":       5088,
        "Pneumonia":    4992,
        "Tuberculosis": 5126,
    },
    "val": {
        "Cancer":        500,
        "COVID-19":      500,
        "Normal":        500,
        "Pneumonia":     500,
        "Tuberculosis":  500,
    },
}


def _scan_split_dir(split_path: Path) -> dict[str, int]:
    if not split_path.exists():
        return {}
    out: dict[str, int] = {}
    for d in sorted(split_path.iterdir()):
        if d.is_dir():
            out[d.name] = sum(
                1 for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS_SCAN
            )
    return out


def load_dataset_counts() -> dict[str, dict[str, int]]:
    """Фактическое число файлов в train/ и val/ (из config)."""
    try:
        import sys
        rp = str(ROOT.resolve())
        if rp not in sys.path:
            sys.path.insert(0, rp)
        from config import TRAIN_DIR, VAL_DIR
        tr = _scan_split_dir(Path(TRAIN_DIR))
        vl = _scan_split_dir(Path(VAL_DIR))
        if tr and vl:
            return {"train": tr, "val": vl}
    except Exception:
        pass
    return {
        "train": dict(DATASET_FALLBACK["train"]),
        "val": dict(DATASET_FALLBACK["val"]),
    }


def _top_wrong_target_row(cm: np.ndarray, names_en: list[str], row_i: int) -> tuple[str, int]:
    """Класс, в который чаще всего уходят примеры истинного row_i (кроме диагонали)."""
    row = cm[row_i]
    best_j, best_n = -1, -1
    for j in range(len(names_en)):
        if j == row_i:
            continue
        if int(row[j]) > best_n:
            best_n, best_j = int(row[j]), j
    return (names_en[best_j], best_n) if best_j >= 0 else ("", 0)


def _cancer_pneumonia_note(data: dict) -> str | None:
    names = data.get("class_names", [])
    cm_list = data.get("confusion_matrix")
    if not names or not cm_list:
        return None
    try:
        i_c = names.index("Cancer")
        i_p = names.index("Pneumonia")
    except ValueError:
        return None
    cm = np.array(cm_list)
    n_cp = int(cm[i_c, i_p])
    n_pc = int(cm[i_p, i_c])
    sup_c = int(data["per_class"]["Cancer"]["support"])
    sup_p = int(data["per_class"]["Pneumonia"]["support"])
    f1_c = float(data["per_class"]["Cancer"]["f1-score"])
    return (
        f"Cancer ↔ Pneumonia:\n"
        f"Cancer→Pneumonia: {n_cp}/{sup_c}\n"
        f"Pneumonia→Cancer: {n_pc}/{sup_p}\n"
        f"→ схожая рентгенкартина.\n"
        f"F1 (Cancer) = {f1_c*100:.1f}%"
    )


def load_metrics() -> dict:
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_history() -> dict:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
def draw_confusion_matrices(ax_abs: plt.Axes, ax_norm: plt.Axes, data: dict):
    cm       = np.array(data["confusion_matrix"])
    names_en = data["class_names"]
    names_ru = [CLASSES_RU[n] for n in names_en]
    cm_norm  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    for ax, mat, fmt, title in [
        (ax_abs,  cm,      "d",    "Матрица ошибок — абсолютные значения"),
        (ax_norm, cm_norm, ".2f",  "Матрица ошибок — нормализованная (Recall per row)"),
    ]:
        ax.set_facecolor(SURFACE)
        cmap = sns.light_palette(BLUE2, as_cmap=True)
        sns.heatmap(
            mat, ax=ax,
            annot=True, fmt=fmt,
            cmap=cmap,
            linewidths=0.6, linecolor=BG,
            xticklabels=names_ru,
            yticklabels=names_ru,
            annot_kws={"size": 9, "weight": "bold", "color": TEXT},
            cbar=True,
        )
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Предсказанный класс", color=SUBTEXT, fontsize=8)
        ax.set_ylabel("Истинный класс",      color=SUBTEXT, fontsize=8)
        ax.tick_params(colors=SUBTEXT, labelsize=7.5)
        ax.xaxis.set_tick_params(rotation=20)
        ax.yaxis.set_tick_params(rotation=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(BLUE)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=SUBTEXT, labelsize=7)
        cbar.outline.set_edgecolor(BORDER)
        # Подсветить диагональ
        for i in range(len(names_en)):
            ax.add_patch(Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor=GREEN, linewidth=2, zorder=5
            ))


# ─────────────────────────────────────────────────────────────────────────────
def draw_metrics_bar(ax: plt.Axes, data: dict):
    names_en = data["class_names"]
    pc       = data["per_class"]

    metrics_keys   = ["precision", "recall", "f1-score"]
    metrics_labels = ["Precision", "Recall", "F1-score"]
    bar_colors     = [BLUE2, GREEN, ORANGE]

    x = np.arange(len(names_en))
    w = 0.22

    for i, (key, lbl, col) in enumerate(zip(metrics_keys, metrics_labels, bar_colors)):
        vals   = [pc.get(c, {}).get(key, 0) for c in names_en]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=lbl, color=col, alpha=0.88,
                        edgecolor=BG, linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{v:.2f}",
                    ha="center", va="bottom",
                    color=col, fontsize=7.5, fontweight="bold",
                )

    _ax_style(ax, "Метрики классификации по классам (Precision / Recall / F1)",
              xlabel="Класс", ylabel="Значение (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CLASSES_RU[n]}\n(n={int(pc.get(n,{}).get('support',0)):,})" for n in names_en],
        fontsize=8.5, color=TEXT,
    )
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(framealpha=0.2, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=9, loc="upper right")

    # Сводная строка
    macro    = data.get("macro", {})
    weighted = data.get("weighted", {})
    summary  = (
        f"Macro avg  →  P: {macro.get('precision',0):.3f}  "
        f"R: {macro.get('recall',0):.3f}  "
        f"F1: {macro.get('f1',0):.3f}      "
        f"Weighted avg  →  P: {weighted.get('precision',0):.3f}  "
        f"R: {weighted.get('recall',0):.3f}  "
        f"F1: {weighted.get('f1',0):.3f}      "
        f"Accuracy: {data.get('accuracy',0)*100:.2f}%"
    )
    ax.text(
        0.5, -0.20, summary,
        transform=ax.transAxes, ha="center", va="bottom",
        color=SUBTEXT, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=BORDER, alpha=0.9),
    )

    # Аннотация Cancer ↔ Pneumonia (из матрицы ошибок)
    note = _cancer_pneumonia_note(data)
    if note:
        cancer_idx = names_en.index("Cancer")
        ax.annotate(
            note,
            xy=(cancer_idx - 0.22, 0.04),
            xytext=(cancer_idx + 0.9, 0.50),
            fontsize=7.5, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=ORANGE, alpha=0.9),
        )


# ─────────────────────────────────────────────────────────────────────────────
def draw_dataset_distribution(ax: plt.Axes, counts: dict[str, dict[str, int]]):
    tr, vl = counts["train"], counts["val"]
    all_names = {*tr.keys(), *vl.keys()}
    names_en = [c for c in CLASS_ORDER if c in all_names] + sorted(
        all_names - set(CLASS_ORDER)
    )
    train_vals = [tr.get(n, 0) for n in names_en]
    val_vals = [vl.get(n, 0) for n in names_en]
    names_ru   = [CLASSES_RU[n] for n in names_en]

    total_train = sum(train_vals)
    total_val   = sum(val_vals)

    x = np.arange(len(names_en))
    w = 0.35

    bars_tr = ax.bar(x - w / 2, train_vals, w,
                     label=f"Train ({total_train:,} снимков)",
                     color=[CLASS_COLORS[n] for n in names_en],
                     alpha=0.88, edgecolor=BG, linewidth=0.5, zorder=3)
    bars_vl = ax.bar(x + w / 2, val_vals, w,
                     label=f"Val ({total_val:,} снимков)",
                     color=[CLASS_COLORS[n] for n in names_en],
                     alpha=0.40, edgecolor=BG, linewidth=0.5,
                     hatch="//", zorder=3)

    for bar, v in zip(bars_tr, train_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:,}", ha="center", va="bottom",
                color=TEXT, fontsize=8.5, fontweight="bold")
    for bar, v in zip(bars_vl, val_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:,}", ha="center", va="bottom",
                color=SUBTEXT, fontsize=8.5)

    _ax_style(ax, "Распределение датасета по классам — реальные данные",
              xlabel="Класс", ylabel="Количество снимков")
    ax.set_xticks(x)
    ax.set_xticklabels(names_ru, fontsize=9.5, color=TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(framealpha=0.2, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=9)

    # Итоговые проценты по классам (train)
    for i, (bar, v) in enumerate(zip(bars_tr, train_vals)):
        pct = v / total_train * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="center", va="center",
                color=BG, fontsize=7.5, fontweight="bold", alpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
def draw_training_history(ax_acc: plt.Axes, ax_loss: plt.Axes, hist: dict):
    if not hist:
        for ax in (ax_acc, ax_loss):
            ax.set_facecolor(SURFACE)
            ax.text(0.5, 0.5, "История обучения недоступна",
                    ha="center", va="center", color=SUBTEXT, fontsize=11,
                    transform=ax.transAxes)
        return

    epochs     = list(range(1, len(hist["train_acc"]) + 1))
    train_acc  = [v * 100 for v in hist["train_acc"]]
    val_acc    = [v * 100 for v in hist["val_acc"]]
    train_loss = hist["train_loss"]
    val_loss   = hist["val_loss"]
    best_epoch = val_acc.index(max(val_acc)) + 1

    # Accuracy
    _ax_style(ax_acc, "Динамика обучения — Accuracy (на обучающей выборке)",
              xlabel="Эпоха", ylabel="Accuracy (%)")
    ax_acc.plot(epochs, train_acc, color=BLUE2, lw=2.2, marker="o", ms=5,
                label="Train Accuracy", zorder=4)
    ax_acc.plot(epochs, val_acc, color=GREEN, lw=2.2, marker="s", ms=5,
                label="Val Accuracy*", zorder=4, ls="--")
    ax_acc.fill_between(epochs, train_acc, alpha=0.08, color=BLUE2)
    ax_acc.fill_between(epochs, val_acc,   alpha=0.08, color=GREEN)
    ax_acc.axvline(best_epoch, color=ORANGE, lw=1.2, ls=":", alpha=0.8)
    ax_acc.text(best_epoch + 0.15, min(train_acc) + 0.5,
                f"best\nval={max(val_acc):.1f}%",
                color=ORANGE, fontsize=7.5)
    ax_acc.set_ylim(min(min(train_acc), min(val_acc)) - 2, 103)
    ax_acc.set_xticks(epochs[::2])
    ax_acc.legend(framealpha=0.2, facecolor=SURFACE, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
    ax_acc.text(
        0.02, 0.07,
        "* Val — та же сбалансированная выборка, что и в evaluate.py\n"
        "  (2 500 снимков, по 500 на класс). Train-линия ниже из‑за аугментации.",
        transform=ax_acc.transAxes, fontsize=7.5, color=SUBTEXT,
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=BORDER, alpha=0.85),
    )

    # Loss
    _ax_style(ax_loss, "Динамика обучения — CrossEntropy Loss",
              xlabel="Эпоха", ylabel="Loss")
    ax_loss.plot(epochs, train_loss, color=ORANGE, lw=2.2, marker="o", ms=5,
                 label="Train Loss", zorder=4)
    ax_loss.plot(epochs, val_loss, color=RED, lw=2.2, marker="s", ms=5,
                 label="Val Loss*", zorder=4, ls="--")
    ax_loss.fill_between(epochs, train_loss, alpha=0.08, color=ORANGE)
    ax_loss.fill_between(epochs, val_loss,   alpha=0.08, color=RED)
    ax_loss.axvline(best_epoch, color=ORANGE, lw=1.2, ls=":", alpha=0.8)
    ax_loss.set_xticks(epochs[::2])
    ax_loss.legend(framealpha=0.2, facecolor=SURFACE, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
def draw_interpretation(ax: plt.Axes, data: dict, counts: dict[str, dict[str, int]]):
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title("Интерпретация результатов оценки модели",
                 color=TEXT, fontsize=11, fontweight="bold", pad=8)

    acc      = data["accuracy"]
    macro    = data["macro"]
    weighted = data["weighted"]
    pc       = data["per_class"]
    names_en = data["class_names"]

    cm = np.array(data.get("confusion_matrix", []))

    rows = []
    for name in names_en:
        col = CLASS_COLORS.get(name, BLUE2)
        pr = pc[name]
        p = float(pr["precision"])
        r = float(pr["recall"])
        f1v = float(pr["f1-score"])
        sup = int(pr["support"])
        row_i = names_en.index(name)
        wrong_cls, wrong_n = _top_wrong_target_row(cm, names_en, row_i)
        wrong_ru = CLASSES_RU.get(wrong_cls, wrong_cls)
        if wrong_n > 0 and sup > 0:
            comment = (
                f"Главная путаница с «{wrong_ru}»: {wrong_n} сн. "
                f"({100.0 * wrong_n / sup:.1f}% выборки класса)."
            )
        else:
            comment = "По матрице ошибок вне диагонали значения малы."
        rows.append((
            name, col,
            f"{p * 100:.1f}%", f"{r * 100:.1f}%", f"{f1v:.3f}", f"{sup:,}",
            comment,
        ))

    col_x = [0.01, 0.22, 0.34, 0.44, 0.53, 0.62, 0.68]
    hdrs  = ["Precision", "Recall", "F1", "Support", "Комментарий"]
    ax.text(col_x[0], 0.93, "Класс", transform=ax.transAxes,
            color=SUBTEXT, fontsize=8.5, fontweight="bold", va="top")
    for hdr, x in zip(hdrs, col_x[2:]):
        ax.text(x, 0.93, hdr, transform=ax.transAxes,
                color=SUBTEXT, fontsize=8.5, fontweight="bold", va="top")

    # Горизонтальная линия-разделитель
    ax.plot([0, 1], [0.88, 0.88], transform=ax.transAxes,
            color=BORDER, lw=0.8, solid_capstyle="round")

    for i, (name, col, prec, rec, f1, support, comment) in enumerate(rows):
        y = 0.79 - i * 0.135
        ax.text(col_x[0], y, f"● {CLASSES_RU[name]}", transform=ax.transAxes,
                color=col, fontsize=9.5, fontweight="bold", va="center")
        for val, x in zip([prec, rec, f1, support, comment],
                           col_x[2:]):
            ax.text(x, y, val, transform=ax.transAxes,
                    color=(SUBTEXT if val == comment else col),
                    fontsize=9 if val != comment else 8,
                    fontfamily="monospace" if val != comment else "DejaVu Sans",
                    va="center")

    # Итоговый блок
    ax.add_patch(FancyBboxPatch(
        (0.0, -0.02), 1.0, 0.15,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01",
        facecolor=BG, edgecolor=BLUE, linewidth=1.2, clip_on=False,
    ))
    ax.text(0.5, 0.08,
            f"Accuracy = {acc*100:.2f}%  |  Macro F1 = {macro['f1']*100:.2f}%  |  "
            f"Weighted F1 = {weighted['f1']*100:.2f}%",
            transform=ax.transAxes, ha="center", va="center",
            color=TEXT, fontsize=11, fontweight="bold")
    n_val = sum(counts["val"].values())
    n_tr = sum(counts["train"].values())
    ax.text(0.5, 0.01,
            f"Оценка на val: {n_val:,} снимков  ·  Train: {n_tr:,} снимков  ·  "
            "20 эпох  ·  ResNet18  ·  Weighted CrossEntropyLoss",
            transform=ax.transAxes, ha="center", va="center",
            color=SUBTEXT, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    data = load_metrics()
    hist = load_history()
    counts = load_dataset_counts()

    # Добавляем историю обучения в data (если есть)
    if hist:
        data["training_history"] = hist

    n_tr = sum(counts["train"].values())
    n_val = sum(counts["val"].values())

    # ── Компоновка: 4 строки ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 30), facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=0.52, wspace=0.35,
        height_ratios=[1.1, 1.0, 0.80, 0.80],
        left=0.06, right=0.97, top=0.955, bottom=0.03,
    )

    ax_cm_abs  = fig.add_subplot(gs[0, 0])
    ax_cm_norm = fig.add_subplot(gs[0, 1])
    ax_bar     = fig.add_subplot(gs[1, :])
    ax_dataset = fig.add_subplot(gs[2, :])
    ax_acc     = fig.add_subplot(gs[3, 0])
    ax_loss    = fig.add_subplot(gs[3, 1])

    draw_confusion_matrices(ax_cm_abs, ax_cm_norm, data)
    draw_metrics_bar(ax_bar, data)
    draw_dataset_distribution(ax_dataset, counts)
    draw_training_history(ax_acc, ax_loss, hist)

    fig.text(0.5, 0.978,
             "LungDx Pro — Полный отчёт качества модели ResNet18",
             ha="center", va="top", color=TEXT, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.965,
             f"Train: {n_tr:,} снимков  ·  Val: {n_val:,} снимков  ·  "
             "20 эпох  ·  NVIDIA RTX 4060  ·  CosineAnnealingLR  ·  Weighted CrossEntropyLoss",
             ha="center", va="top", color=SUBTEXT, fontsize=9.5)

    out1 = OUT_DIR / "metrics_full_report.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Сохранено: {out1}")

    # ── Отдельная интерпретационная карта ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(18, 5.5), facecolor=BG)
    ax2.set_facecolor(BG)
    draw_interpretation(ax2, data, counts)
    fig2.text(0.5, 0.99,
              "Интерпретация метрик классификации — LungDx Pro",
              ha="center", va="top", color=TEXT, fontsize=13, fontweight="bold")
    out2 = OUT_DIR / "metrics_interpretation.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"Сохранено: {out2}")


if __name__ == "__main__":
    main()
