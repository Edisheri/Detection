"""
Перераспределение датасета — сбалансированный Train/Val split.

Алгоритм:
  1. Собирает ВСЕ файлы класса из train/ и val/ в один список.
  2. Случайный shuffle (seed фиксирован).
  3. Первые VAL_PER_CLASS → val/, остальные → train/.
  4. Перемещает только те файлы которые ещё не в нужной папке.
  5. Дублирующиеся имена разрешаются числовым суффиксом.

Запуск:
    .venv\\Scripts\\python.exe scripts/rebalance_dataset.py
    .venv\\Scripts\\python.exe scripts/rebalance_dataset.py --val-per-class 400 --yes
"""
import argparse
import random
import shutil
from pathlib import Path

RANDOM_SEED   = 42
VAL_PER_CLASS = 500

ROOT      = Path(__file__).resolve().parent.parent
XRAY_DIR  = ROOT / "data" / "data" / "chest_xray"
TRAIN_DIR = XRAY_DIR / "train"
VAL_DIR   = XRAY_DIR / "val"
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_all(cls: str) -> list[Path]:
    files = []
    for split_dir in (TRAIN_DIR, VAL_DIR):
        d = split_dir / cls
        if d.exists():
            files += [p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files


def move_to(src: Path, dst_dir: Path) -> None:
    """Перемещает src в dst_dir. Если уже там — пропускает. Разрешает конфликт имён."""
    if src.parent.resolve() == dst_dir.resolve():
        return   # уже на месте
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = src.stem, src.suffix
        i = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{i}{suf}"
            i += 1
    shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--yes", action="store_true", help="Без подтверждения")
    args = parser.parse_args()

    random.seed(args.seed)

    classes = sorted(set(
        [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()] +
        [d.name for d in VAL_DIR.iterdir()   if d.is_dir()]
    ))

    print(f"\nКлассы: {classes}")
    print(f"Val per class: {args.val_per_class}\n")
    print(f"{'Класс':<16} {'Всего':>7} {'Train':>8} {'Val':>6}")
    print("-" * 42)

    plan: dict[str, tuple[list[Path], list[Path]]] = {}
    for cls in classes:
        files = collect_all(cls)
        random.shuffle(files)
        n_val = min(args.val_per_class, len(files) // 5)
        plan[cls] = (files[n_val:], files[:n_val])   # (train_files, val_files)
        print(f"{cls:<16} {len(files):>7,} {len(files)-n_val:>8,} {n_val:>6,}")

    print()
    if not args.yes:
        confirm = input("Продолжить? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Отменено.")
            return

    for cls, (train_files, val_files) in plan.items():
        train_dst = TRAIN_DIR / cls
        val_dst   = VAL_DIR   / cls
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        # Перемещаем val-файлы сначала (могут быть в train — двигаем их в val)
        for f in val_files:
            move_to(f, val_dst)

        # Перемещаем train-файлы (могут быть в val — двигаем их в train)
        for f in train_files:
            move_to(f, train_dst)

        n_tr = len([p for p in train_dst.iterdir() if p.is_file()])
        n_vl = len([p for p in val_dst.iterdir()   if p.is_file()])
        print(f"  {cls}: train={n_tr:,}  val={n_vl:,}  OK")

    print("\nИтог:")
    print(f"{'Класс':<16} {'Train':>8} {'Val':>6}")
    print("-" * 32)
    total_tr = total_vl = 0
    for cls in classes:
        n_tr = len([p for p in (TRAIN_DIR / cls).iterdir() if p.is_file()])
        n_vl = len([p for p in (VAL_DIR   / cls).iterdir() if p.is_file()])
        total_tr += n_tr
        total_vl += n_vl
        print(f"{cls:<16} {n_tr:>8,} {n_vl:>6,}")
    print("-" * 32)
    print(f"{'ИТОГО':<16} {total_tr:>8,} {total_vl:>6,}")
    print(f"\nГотово! Теперь запустите: .venv\\Scripts\\python.exe train.py")


if __name__ == "__main__":
    main()
