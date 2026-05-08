"""
Генерация диаграмм для магистерской работы LungDx Pro.

Создаёт:
  - confusion_matrix.png       — матрица ошибок модели
  - diagram_idef0.png          — IDEF0 контекстная + декомпозиция
  - diagram_usecase.png        — диаграмма прецедентов
  - diagram_classes.png        — диаграмма классов (UML)
  - diagram_architecture.png   — архитектура ResNet18
  - training_history.png       — кривые обучения

Запуск: python scripts/generate_diagrams.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import seaborn as sns

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "reports" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── Палитра (медицинский тёмно-синий стиль) ──────────────────────────────────
C_BG        = "#0D1117"   # фон страницы
C_SURFACE   = "#161B22"   # поверхность блоков
C_BORDER    = "#1F6FEB"   # акцент синий
C_ACCENT2   = "#388BFD"   # вторичный акцент
C_GREEN     = "#3FB950"   # успех / ОК
C_RED       = "#F85149"   # ошибка
C_YELLOW    = "#D29922"   # предупреждение
C_TEXT      = "#E6EDF3"   # основной текст
C_SUBTEXT   = "#8B949E"   # второстепенный текст
C_HIGHLIGHT = "#1F6FEB22" # полупрозрачная заливка

FONT_FAMILY = "DejaVu Sans"

def _fig(w=14, h=9):
    fig = plt.figure(figsize=(w, h), facecolor=C_BG)
    return fig

def _box(ax, x, y, w, h, label, sublabel="", color=C_BORDER, fs=10, lw=2):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02",
                           linewidth=lw, edgecolor=color,
                           facecolor=C_SURFACE, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                color=C_TEXT, fontsize=fs, fontweight="bold",
                fontfamily=FONT_FAMILY, zorder=4)
        ax.text(x + w/2, y + h*0.28, sublabel, ha="center", va="center",
                color=C_SUBTEXT, fontsize=fs-2,
                fontfamily=FONT_FAMILY, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=C_TEXT, fontsize=fs, fontweight="bold",
                fontfamily=FONT_FAMILY, zorder=4)

def _arrow(ax, x1, y1, x2, y2, label="", color=C_SUBTEXT):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha="center", va="center",
                color=C_SUBTEXT, fontsize=8,
                fontfamily=FONT_FAMILY,
                bbox=dict(boxstyle="round,pad=0.2", fc=C_BG, ec="none"),
                zorder=5)

def _title(ax, text, sub=""):
    ax.text(0.5, 0.97, text, transform=ax.transAxes,
            ha="center", va="top", color=C_TEXT,
            fontsize=14, fontweight="bold", fontfamily=FONT_FAMILY)
    if sub:
        ax.text(0.5, 0.93, sub, transform=ax.transAxes,
                ha="center", va="top", color=C_SUBTEXT,
                fontsize=9, fontfamily=FONT_FAMILY)

# ─────────────────────────────────────────────────────────────────────────────
# 1. МАТРИЦА ОШИБОК
# ─────────────────────────────────────────────────────────────────────────────
def draw_confusion_matrix():
    metrics_path = ROOT / "reports" / "metrics_summary.json"
    if not metrics_path.exists():
        print("  [!] metrics_summary.json не найден, пропускаем")
        return

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    cm = np.array(m["confusion_matrix"])
    classes_en = m.get("class_names", ["COVID-19","Cancer","Normal","Pneumonia","Tuberculosis"])
    classes_ru = ["COVID-19", "Рак\nлёгкого", "Норма", "Пневмония", "Туберкулёз"]

    # нормализованная матрица (по строкам)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_BG)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Абсолютные значения", "Нормализованная (по строкам)"],
        [".0f", ".2f"]
    ):
        ax.set_facecolor(C_BG)
        cmap = sns.light_palette(C_BORDER, as_cmap=True)
        sns.heatmap(
            data, ax=ax,
            annot=True, fmt=fmt, cmap=cmap,
            linewidths=0.5, linecolor=C_BG,
            cbar=True,
            xticklabels=classes_ru,
            yticklabels=classes_ru,
            annot_kws={"size": 11, "color": C_TEXT, "weight": "bold"},
        )
        ax.set_title(title, color=C_TEXT, fontsize=12,
                     fontweight="bold", pad=12, fontfamily=FONT_FAMILY)
        ax.set_xlabel("Предсказанный класс", color=C_SUBTEXT,
                      fontsize=10, fontfamily=FONT_FAMILY)
        ax.set_ylabel("Истинный класс", color=C_SUBTEXT,
                      fontsize=10, fontfamily=FONT_FAMILY)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.set_tick_params(rotation=0)
        ax.yaxis.set_tick_params(rotation=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=C_SUBTEXT, labelsize=8)

    fig.suptitle("Матрица ошибок — LungDx Pro (ResNet18)",
                 color=C_TEXT, fontsize=15, fontweight="bold",
                 fontfamily=FONT_FAMILY, y=1.02)
    fig.text(0.5, -0.02,
             "Валидационная выборка: 709 снимков  |  Accuracy = 98.87%",
             ha="center", color=C_SUBTEXT, fontsize=9, fontfamily=FONT_FAMILY)

    plt.tight_layout()
    out = OUT / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. КРИВЫЕ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────
def draw_training_history():
    hist_path = ROOT / "weights" / "training_history.json"
    if not hist_path.exists():
        print("  [!] training_history.json не найден")
        return

    with open(hist_path, encoding="utf-8") as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_acc"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.tick_params(colors=C_SUBTEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_BORDER)
            spine.set_alpha(0.4)
        ax.grid(True, color=C_BORDER, alpha=0.15, linestyle="--")

    # Accuracy
    ax1.plot(epochs, [a*100 for a in h["train_acc"]],
             color=C_ACCENT2, lw=2, marker="o", ms=4, label="Train Accuracy")
    if h.get("val_acc"):
        ax1.plot(epochs, [a*100 for a in h["val_acc"]],
                 color=C_GREEN, lw=2, marker="s", ms=4, label="Val Accuracy")
    ax1.set_xlabel("Эпоха", color=C_SUBTEXT, fontsize=10)
    ax1.set_ylabel("Accuracy (%)", color=C_SUBTEXT, fontsize=10)
    ax1.set_title("Точность по эпохам", color=C_TEXT, fontsize=12,
                  fontweight="bold", fontfamily=FONT_FAMILY)
    ax1.legend(framealpha=0.2, facecolor=C_SURFACE, edgecolor=C_BORDER,
               labelcolor=C_TEXT, fontsize=9)
    ax1.set_ylim(60, 102)

    # Loss
    ax2.plot(epochs, h["train_loss"],
             color=C_ACCENT2, lw=2, marker="o", ms=4, label="Train Loss")
    if h.get("val_loss"):
        ax2.plot(epochs, h["val_loss"],
                 color=C_GREEN, lw=2, marker="s", ms=4, label="Val Loss")
    ax2.set_xlabel("Эпоха", color=C_SUBTEXT, fontsize=10)
    ax2.set_ylabel("Loss", color=C_SUBTEXT, fontsize=10)
    ax2.set_title("Функция потерь по эпохам", color=C_TEXT, fontsize=12,
                  fontweight="bold", fontfamily=FONT_FAMILY)
    ax2.legend(framealpha=0.2, facecolor=C_SURFACE, edgecolor=C_BORDER,
               labelcolor=C_TEXT, fontsize=9)

    fig.suptitle("История обучения ResNet18 — LungDx Pro",
                 color=C_TEXT, fontsize=14, fontweight="bold",
                 fontfamily=FONT_FAMILY)
    plt.tight_layout()
    out = OUT / "training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. IDEF0
# ─────────────────────────────────────────────────────────────────────────────
def draw_idef0():
    fig = _fig(16, 10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off"); ax.set_facecolor(C_BG)

    _title(ax, "IDEF0 — Декомпозиция системы LungDx Pro",
           "Уровень A0: Автоматическая диагностика лёгочных заболеваний")

    # ── Центральный блок A0 ──
    _box(ax, 5.5, 3.8, 5, 2.4,
         "A0  ДИАГНОСТИКА\nЛЁГОЧНЫХ ЗАБОЛЕВАНИЙ",
         color=C_BORDER, fs=12, lw=2.5)

    # Входы (слева)
    inputs = [
        (0.3, 6.4, "Рентгеновский снимок"),
        (0.3, 5.2, "Параметры порогов"),
        (0.3, 4.0, "Клинические регламенты"),
    ]
    for x, y, lbl in inputs:
        _box(ax, x, y-0.3, 2.4, 0.55, lbl, color=C_SUBTEXT, fs=8, lw=1)
        _arrow(ax, x+2.4, y, 5.5, y, color=C_ACCENT2)

    # Выходы (справа)
    outputs = [
        (13.3, 6.4, "Диагноз + уверенность"),
        (13.3, 5.2, "PDF / JSON отчёт"),
        (13.3, 4.0, "Аудит-лог решений"),
    ]
    for x, y, lbl in outputs:
        _box(ax, x, y-0.3, 2.4, 0.55, lbl, color=C_GREEN, fs=8, lw=1)
        _arrow(ax, 10.5, y, x, y, color=C_GREEN)

    # Управление (сверху)
    _box(ax, 5.5, 7.6, 5, 0.6, "Медицинские стандарты диагностики", color=C_YELLOW, fs=8, lw=1)
    _arrow(ax, 8.0, 7.6, 8.0, 6.2, color=C_YELLOW)

    # Механизм (снизу)
    _box(ax, 5.5, 2.2, 5, 0.6, "ResNet18 + GPU NVIDIA RTX 4060 + Python", color=C_SUBTEXT, fs=8, lw=1)
    _arrow(ax, 8.0, 3.8, 8.0, 2.8, color=C_SUBTEXT)

    # ── Декомпозиция (нижняя часть) ──
    blocks = [
        (0.4,  0.2, 2.8, 1.6, "A1\nКОНТРОЛЬ\nКАЧЕСТВА",     "X-Ray Score\nяркость / контраст", C_YELLOW),
        (3.4,  0.2, 2.8, 1.6, "A2\nПРЕДОБРА-\nБОТКА",         "Resize 224×224\nнормализация",    C_ACCENT2),
        (6.4,  0.2, 2.8, 1.6, "A3\nКЛАССИФИ-\nКАЦИЯ",         "ResNet18\nSoftmax → 5 кл.",       C_BORDER),
        (9.4,  0.2, 2.8, 1.6, "A4\nGRAD-CAM",                  "Карта внимания\nBackprop",         C_GREEN),
        (12.4, 0.2, 2.8, 1.6, "A5\nОТЧЁТ",                     "PDF / JSON\nРекомендации",         C_SUBTEXT),
    ]
    for i, (x, y, w, h, lbl, sub, col) in enumerate(blocks):
        _box(ax, x, y, w, h, lbl, sublabel=sub, color=col, fs=9, lw=1.5)
        if i < len(blocks)-1:
            _arrow(ax, x+w, y+h/2, x+w+0.6, y+h/2, color=C_SUBTEXT)

    ax.text(8, 0.02, "Декомпозиция A0 → функциональные блоки A1–A5",
            ha="center", va="bottom", color=C_SUBTEXT, fontsize=8,
            fontfamily=FONT_FAMILY)

    ax.plot([0, 16], [2.0, 2.0], color=C_BORDER, alpha=0.3, lw=0.8, ls="--")

    out = OUT / "diagram_idef0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. USE CASE
# ─────────────────────────────────────────────────────────────────────────────
def _actor(ax, x, y, label, color=C_ACCENT2):
    """Рисует фигурку актора."""
    # голова
    circle = plt.Circle((x, y+0.55), 0.18, color=color, zorder=4)
    ax.add_patch(circle)
    # тело
    ax.plot([x, x], [y+0.37, y+0.05], color=color, lw=2, zorder=4)
    # руки
    ax.plot([x-0.3, x+0.3], [y+0.25, y+0.25], color=color, lw=2, zorder=4)
    # ноги
    ax.plot([x, x-0.25], [y+0.05, y-0.25], color=color, lw=2, zorder=4)
    ax.plot([x, x+0.25], [y+0.05, y-0.25], color=color, lw=2, zorder=4)
    ax.text(x, y-0.42, label, ha="center", va="top",
            color=color, fontsize=8.5, fontweight="bold",
            fontfamily=FONT_FAMILY, zorder=4)


def _usecase(ax, x, y, w, h, text, color=C_BORDER):
    ellipse = mpatches.Ellipse((x, y), w, h,
                                linewidth=1.5, edgecolor=color,
                                facecolor=C_SURFACE, zorder=3)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha="center", va="center",
            color=C_TEXT, fontsize=7.5, fontfamily=FONT_FAMILY,
            zorder=4, wrap=True,
            multialignment="center")


def draw_usecase():
    fig = _fig(16, 11)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.axis("off"); ax.set_facecolor(C_BG)

    _title(ax, "Диаграмма прецедентов — LungDx Pro",
           "Варианты использования системы")

    # граница системы
    sys_rect = FancyBboxPatch((2.5, 0.4), 11, 9.8,
                               boxstyle="round,pad=0.1",
                               linewidth=1.5, edgecolor=C_BORDER,
                               facecolor=C_HIGHLIGHT, zorder=1, alpha=0.5)
    ax.add_patch(sys_rect)
    ax.text(8.0, 10.1, "«Система» LungDx Pro", ha="center",
            color=C_ACCENT2, fontsize=10, fontweight="bold",
            fontfamily=FONT_FAMILY)

    # Акторы
    _actor(ax, 0.9, 7.5, "Врач-\nрентгенолог", C_ACCENT2)
    _actor(ax, 15.1, 7.5, "Администратор", C_YELLOW)
    _actor(ax, 0.9, 2.5, "ResNet18\n(нейросеть)", C_GREEN)

    # Прецеденты врача
    doctor_ucs = [
        (5.5, 9.0, "Загрузить\nснимок"),
        (5.5, 7.4, "Просмотреть\nдиагноз"),
        (5.5, 5.8, "Тепловая карта\nGrad-CAM"),
        (5.5, 4.2, "Скачать\nPDF/JSON отчёт"),
        (5.5, 2.6, "Пакетный\nанализ"),
        (5.5, 1.0, "Рабочая\nочередь"),
    ]
    for x, y, t in doctor_ucs:
        _usecase(ax, x, y, 3.2, 0.75, t, C_ACCENT2)
        _arrow(ax, 1.4, 8.1, x-1.6, y+0.1, color=C_ACCENT2)

    # Прецеденты администратора
    admin_ucs = [
        (10.5, 9.0, "Настроить порог\nавто-решения"),
        (10.5, 7.4, "Model Card /\nТехн. паспорт"),
        (10.5, 5.8, "Экспорт\nаудит-лога"),
    ]
    for x, y, t in admin_ucs:
        _usecase(ax, x, y, 3.2, 0.75, t, C_YELLOW)
        _arrow(ax, 14.6, 8.1, x+1.6, y+0.1, color=C_YELLOW)

    # Системные прецеденты
    sys_ucs = [
        (10.5, 4.2, "Фильтр\nне-рентгеновских"),
        (10.5, 2.6, "Контроль\nкачества снимка"),
        (10.5, 1.0, "Аудит-лог\nдействий"),
    ]
    for x, y, t in sys_ucs:
        _usecase(ax, x, y, 3.2, 0.75, t, C_GREEN)
        _arrow(ax, 1.4, 3.0, x-1.6, y+0.1, color=C_GREEN)

    out = OUT / "diagram_usecase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ДИАГРАММА КЛАССОВ (UML)
# ─────────────────────────────────────────────────────────────────────────────
def _uml_class(ax, x, y, w, class_name, attrs, methods, color=C_BORDER):
    line_h = 0.28
    header_h = 0.5
    section_h = max(len(attrs), 1) * line_h + 0.1
    method_h  = max(len(methods), 1) * line_h + 0.1
    total_h   = header_h + section_h + method_h

    # header
    header = FancyBboxPatch((x, y - header_h), w, header_h,
                             boxstyle="round,pad=0.01",
                             linewidth=1.5, edgecolor=color,
                             facecolor=color + "55", zorder=3)
    ax.add_patch(header)
    ax.text(x + w/2, y - header_h/2, class_name,
            ha="center", va="center", color=C_TEXT,
            fontsize=8, fontweight="bold", fontfamily=FONT_FAMILY, zorder=4)

    # attrs section
    attrs_rect = FancyBboxPatch((x, y - header_h - section_h), w, section_h,
                                 boxstyle="square,pad=0",
                                 linewidth=1.5, edgecolor=color,
                                 facecolor=C_SURFACE, zorder=3)
    ax.add_patch(attrs_rect)
    for i, a in enumerate(attrs):
        ax.text(x + 0.08, y - header_h - 0.08 - i*line_h, a,
                ha="left", va="top", color=C_SUBTEXT,
                fontsize=6.5, fontfamily=FONT_FAMILY, zorder=4)

    # methods section
    meth_rect = FancyBboxPatch((x, y - total_h), w, method_h,
                                boxstyle="square,pad=0",
                                linewidth=1.5, edgecolor=color,
                                facecolor=C_SURFACE, zorder=3)
    ax.add_patch(meth_rect)
    for i, m in enumerate(methods):
        ax.text(x + 0.08, y - header_h - section_h - 0.08 - i*line_h, m,
                ha="left", va="top", color=C_TEXT,
                fontsize=6.5, fontfamily=FONT_FAMILY, zorder=4)

    return x + w/2, y - total_h, x + w/2, y  # cx_bot, cy_bot, cx_top, cy_top


def draw_classes():
    fig = _fig(18, 11)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.axis("off"); ax.set_facecolor(C_BG)

    _title(ax, "Диаграмма классов — LungDx Pro",
           "Основные модули и их взаимосвязи")

    classes = [
        # (x, y, w, name, attrs, methods, color)
        (0.3, 10.5, 3.2, "StreamlitApp",
         ["session_state: Dict", "history: List"],
         ["render_sidebar()", "render_single_analysis()",
          "render_batch_analysis()", "render_worklist()",
          "render_history()"],
         C_BORDER),

        (4.1, 10.5, 3.2, "InferenceEngine",
         ["model: ResNet18", "device: torch.device",
          "class_names: List[str]"],
         ["load_model(path) → ResNet18",
          "predict_image(img) → Dict",
          "generate_gradcam(img, cls) → ndarray"],
         C_ACCENT2),

        (7.9, 10.5, 3.2, "QualityGate",
         ["score: float", "details: Dict"],
         ["is_likely_chest_xray(img) → Dict",
          "_check_brightness(img) → float",
          "_check_contrast(img) → float"],
         C_YELLOW),

        (11.7, 10.5, 3.0, "ReportBuilder",
         [],
         ["build_case_pdf(payload) → bytes",
          "build_session_pdf(hist) → bytes",
          "save_json(payload) → str"],
         C_GREEN),

        (0.3, 6.3, 3.2, "ChestXRayDataset",
         ["root: Path", "samples: List[Tuple]",
          "class_to_idx: Dict", "transform: Compose"],
         ["__len__() → int",
          "__getitem__(idx) → Tuple",
          "_load_samples()"],
         C_SUBTEXT),

        (4.1, 6.3, 3.2, "ResNet18Model",
         ["backbone: ResNet", "fc: Linear(512→5)"],
         ["forward(x: Tensor) → Tensor",
          "build_model(n_cls, pretrained) → Self"],
         C_BORDER),

        (7.9, 6.3, 3.2, "Metrics",
         [],
         ["evaluate_model(model, loader) → Dict",
          "save_classification_report(eval, path)",
          "save_confusion_matrix_png(eval, path)",
          "save_metrics_summary(eval, path)"],
         C_ACCENT2),

        (11.7, 6.3, 3.0, "TrainPipeline",
         ["epochs: int", "lr: float",
          "batch_size: int"],
         ["compute_class_weights() → Tensor",
          "train_epoch() → Tuple",
          "validate() → Tuple"],
         C_YELLOW),
    ]

    centers = {}
    for args in classes:
        x, y, w, name = args[0], args[1], args[2], args[3]
        cx_b, cy_b, cx_t, cy_t = _uml_class(ax, *args)
        centers[name] = {"bot": (cx_b, cy_b), "top": (cx_t, cy_t),
                         "left": (x, y - 0.25), "right": (x+w, y-0.25)}

    # Связи
    relations = [
        ("StreamlitApp",    "InferenceEngine", "использует →"),
        ("StreamlitApp",    "QualityGate",     "использует →"),
        ("StreamlitApp",    "ReportBuilder",   "использует →"),
        ("InferenceEngine", "ResNet18Model",   "содержит →"),
        ("InferenceEngine", "ChestXRayDataset","загружает →"),
        ("Metrics",         "ResNet18Model",   "оценивает →"),
        ("TrainPipeline",   "ChestXRayDataset","обучает на →"),
    ]
    for src, dst, lbl in relations:
        if src in centers and dst in centers:
            x1, y1 = centers[src]["bot"]
            x2, y2 = centers[dst]["top"]
            _arrow(ax, x1, y1, x2, y2, lbl, color=C_SUBTEXT)

    out = OUT / "diagram_classes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. АРХИТЕКТУРА ResNet18
# ─────────────────────────────────────────────────────────────────────────────
def draw_architecture():
    fig = _fig(18, 9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, 9)
    ax.axis("off"); ax.set_facecolor(C_BG)

    _title(ax, "Архитектура нейронной сети ResNet18 (Transfer Learning)",
           "Вход: 3×224×224  →  Выход: 5 классов лёгочных заболеваний")

    layers = [
        # (x, y, w, h, label, sublabel, color)
        (0.3,  3.5, 1.8, 2.0, "Вход",           "3×224×224\nRGB", C_SUBTEXT),
        (2.4,  3.2, 2.2, 2.6, "Conv 7×7\nBN+ReLU\nMaxPool",
                                                  "64×56×56",      C_YELLOW),
        (5.0,  3.2, 2.2, 2.6, "Layer 1\n×2 блока\nResidual",
                                                  "64×56×56",      C_ACCENT2),
        (7.6,  3.2, 2.2, 2.6, "Layer 2\n×2 блока\nResidual",
                                                  "128×28×28",     C_ACCENT2),
        (10.2, 3.2, 2.2, 2.6, "Layer 3\n×2 блока\nResidual",
                                                  "256×14×14",     C_ACCENT2),
        (12.8, 3.2, 2.2, 2.6, "Layer 4\n×2 блока\nResidual",
                                                  "512×7×7",       C_BORDER),
        (15.4, 3.5, 1.3, 2.0, "AvgPool\nFlatten", "512",          C_BORDER),
    ]

    prev_x = None
    for x, y, w, h, lbl, sub, col in layers:
        _box(ax, x, y, w, h, lbl, sublabel=sub, color=col, fs=9)
        if prev_x is not None:
            _arrow(ax, prev_x, y + h/2, x, y + h/2, color=C_SUBTEXT)
        prev_x = x + w

    # FC
    fc_x = 15.4
    _box(ax, fc_x, 1.8, 1.3, 1.2, "FC\nLinear\n512→5", color=C_GREEN, fs=8)
    _arrow(ax, fc_x + 0.65, 3.5, fc_x + 0.65, 3.0, color=C_SUBTEXT)

    # Softmax
    _box(ax, fc_x, 0.4, 1.3, 1.0, "Softmax\n5 классов", color=C_GREEN, fs=8)
    _arrow(ax, fc_x + 0.65, 1.8, fc_x + 0.65, 1.4, color=C_GREEN)

    # Residual block схема
    rb_x, rb_y = 1.0, 0.5
    ax.text(rb_x, rb_y + 1.3, "Residual Block:", color=C_TEXT, fontsize=8,
            fontweight="bold", fontfamily=FONT_FAMILY)
    _box(ax, rb_x,     rb_y, 1.2, 0.6, "Conv 3×3\nBN+ReLU", color=C_ACCENT2, fs=7)
    _box(ax, rb_x+1.4, rb_y, 1.2, 0.6, "Conv 3×3\nBN",      color=C_ACCENT2, fs=7)
    _box(ax, rb_x+2.8, rb_y, 0.8, 0.6, "  +\n ReLU",        color=C_GREEN,   fs=7)
    _arrow(ax, rb_x+1.2, rb_y+0.3, rb_x+1.4, rb_y+0.3, color=C_SUBTEXT)
    _arrow(ax, rb_x+2.6, rb_y+0.3, rb_x+2.8, rb_y+0.3, color=C_SUBTEXT)
    # skip
    ax.annotate("", xy=(rb_x+2.8+0.4, rb_y+0.3),
                xytext=(rb_x, rb_y+0.9),
                arrowprops=dict(arrowstyle="-|>", color=C_YELLOW,
                                lw=1.5, connectionstyle="arc3,rad=-0.4"))
    ax.text(rb_x+1.8, rb_y+1.1, "skip connection",
            color=C_YELLOW, fontsize=7, fontfamily=FONT_FAMILY, ha="center")

    # Выходные классы
    class_names = ["COVID-19", "Рак лёгкого", "Норма", "Пневмония", "Туберкулёз"]
    colors_cls  = [C_RED, C_RED, C_GREEN, C_YELLOW, C_YELLOW]
    for i, (cls, col) in enumerate(zip(class_names, colors_cls)):
        cy = 8.2 - i * 1.3
        _box(ax, 14.0, cy - 0.25, 2.2, 0.5, cls, color=col, fs=7.5)
        _arrow(ax, 16.05, 2.4, 14.0 + 1.1, cy, color=col)

    out = OUT / "diagram_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BAR-CHART МЕТРИК ПО КЛАССАМ
# ─────────────────────────────────────────────────────────────────────────────
def draw_metrics_bar():
    metrics_path = ROOT / "reports" / "classification_report.json"
    if not metrics_path.exists():
        print("  [!] classification_report.json не найден")
        return

    with open(metrics_path, encoding="utf-8") as f:
        cr = json.load(f)

    classes_en = ["COVID-19", "Cancer", "Normal", "Pneumonia", "Tuberculosis"]
    classes_ru = ["COVID-19", "Рак лёгкого", "Норма", "Пневмония", "Туберкулёз"]
    metrics    = ["precision", "recall", "f1-score"]
    labels_ru  = ["Точность (Precision)", "Полнота (Recall)", "F1-мера"]
    bar_colors = [C_ACCENT2, C_GREEN, C_BORDER]

    x = np.arange(len(classes_en))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=C_BG)
    ax.set_facecolor(C_SURFACE)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_alpha(0.4)
    ax.grid(True, axis="y", color=C_BORDER, alpha=0.2, linestyle="--")

    for i, (m, lbl, col) in enumerate(zip(metrics, labels_ru, bar_colors)):
        vals = [cr.get(c, {}).get(m, 0) for c in classes_en]
        bars = ax.bar(x + i*width - width, vals, width,
                      label=lbl, color=col, alpha=0.85, zorder=3,
                      edgecolor=C_BG, linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=C_TEXT, fontsize=8, fontfamily=FONT_FAMILY)

    ax.set_xticks(x)
    ax.set_xticklabels(classes_ru, fontsize=10, color=C_TEXT,
                       fontfamily=FONT_FAMILY)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Значение метрики", color=C_SUBTEXT, fontsize=11,
                  fontfamily=FONT_FAMILY)
    ax.set_title("Метрики классификации по классам — LungDx Pro (ResNet18)",
                 color=C_TEXT, fontsize=13, fontweight="bold",
                 fontfamily=FONT_FAMILY, pad=15)
    ax.legend(framealpha=0.2, facecolor=C_SURFACE, edgecolor=C_BORDER,
              labelcolor=C_TEXT, fontsize=10)

    macro = cr.get("macro avg", {})
    fig.text(0.5, 0.01,
             f"Macro avg → Precision: {macro.get('precision',0):.3f}  "
             f"Recall: {macro.get('recall',0):.3f}  "
             f"F1: {macro.get('f1-score',0):.3f}  |  Accuracy: 98.87%",
             ha="center", color=C_SUBTEXT, fontsize=9, fontfamily=FONT_FAMILY)

    plt.tight_layout()
    out = OUT / "metrics_by_class.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Генерация диаграмм → {OUT}\n")
    draw_confusion_matrix()
    draw_training_history()
    draw_idef0()
    draw_usecase()
    draw_classes()
    draw_architecture()
    draw_metrics_bar()
    print(f"\nГотово! Все файлы сохранены в {OUT}")
