from collections import defaultdict
from pathlib import Path

class_names = [
    "uniform_top", "uniform_bottom",
    "civilian_top_short_sleeve", "civilian_top_long_sleeve",
    "civilian_bottom_trousers", "civilian_bottom_shorts", "civilian_bottom_skirt",
    "footwear_shoes", "footwear_slippers",
    "prohibited_ripped_pants", "prohibited_leggings",
    "prohibited_sleeveless", "prohibited_crop_halter", "prohibited_midriff_offshoulder",
]

datasets = {
    "full/train":     Path("dataset/full/labels/train"),
    "balanced/train": Path("dataset/balanced/labels/train"),
    "balanced/val":   Path("dataset/balanced/labels/val"),
    "balanced/test":  Path("dataset/balanced/labels/test"),
}

for split_name, label_dir in datasets.items():
    if not label_dir.exists():
        continue
    files = list(label_dir.glob("*.txt"))
    total_imgs = len(files)
    counts = defaultdict(int)
    for f in files:
        seen = set()
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    seen.add(int(line.split()[0]))
        except Exception:
            pass
        for c in seen:
            counts[c] += 1

    print(f"\n[{split_name}]  ({total_imgs} total images)")
    print(f"  {'ID':<4} {'Class':<38} {'Images':>6}  {'% of split':>10}")
    print(f"  {'-'*56}")
    for i, name in enumerate(class_names):
        n = counts[i]
        pct = n / total_imgs * 100 if total_imgs else 0
        print(f"  {i:<4} {name:<38} {n:>6}  {pct:>9.1f}%")
