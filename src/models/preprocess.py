"""
preprocess.py
-------------
Converts COCO annotations to YOLO format, merges 356 granular
product names into ~15 meaningful categories, splits into
train/val/test, and verifies the dataset.

Usage:
    # Med klassamerging (anbefalt):
    python src/preprocess.py --data_dir res/dataset/train --format coco

    # Uten merging (alle 356 klasser):
    python src/preprocess.py --data_dir res/dataset/train --format coco --no_merge
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
#  Klassamerging — 356 produktnavn → 15 kategorier
# ─────────────────────────────────────────────────────────────

# Nøkkelord som matcher produktnavn (store bokstaver, sjekkes i rekkefølge)
# Første match vinner — legg spesifikke nøkkelord OVER generelle
CLASS_MERGE_RULES = [
    ("kaffe_kapsel",      ["KAPSEL", "KAPSLER", "TASSIMO", "DOLCE GUSTO", "NESCAFE AZERA",
                           "COTW", "AMERICANO 16", "GRANDE INTENSO", "FLAT WHITE 16",
                           "CAFE AU LAIT 16", "ESPRESSO INTENSO 16", "CAPPUCCINO 8"]),
    ("kaffe_helbønne",    ["HELE BØNNER"]),
    ("kaffe_malt",        ["FILTERMALT", "KOKMALT", "PRESSMALT", "KAFFEPUTER",
                           "FILTERPOSER", "KAFFEFILTER"]),
    ("kaffe_instant",     ["INSTANT", "NESCAFE", "NESCAFÉ", "BRASERO", "AZERA"]),
    ("te",                ["TE ", "TEA ", "TEA\n", "CHAI", "ROOIBOS", "PUKKA",
                           "TWININGS", "LIPTON", "URTETE", "KAMILLE", "EARL GREY",
                           "GREEN CEYLON", "ICETEA"]),
    ("kakao_sjokolade",   ["KAKAO", "SJOKOLADEDRIKK", "NESQUIK", "O'BOY", "REGIA KAKAO",
                           "BAKEKAKAO", "RETT I KOPPEN"]),
    ("frokostblanding",   ["FROKOSTBLANDING", "CORN FLAKES", "CHEERIOS", "NESQUIK FROKOST",
                           "FITNESS FROKOST", "WEETABIX", "SMACKS", "COCO POPS",
                           "FROKOST-TALL", "SJOKORINGER", "PUFFET HAVRE", "PUFFET HVETE",
                           "RIS PUFFET", "HAVRERINGER", "HAVREFRAS", "WEETOS",
                           "TRESOR", "OREO O'S", "LION", "ALL-BRAN", "SPECIAL K"]),
    ("granola_müsli",     ["GRANOLA", "MÜSLI", "MUSLI", "MUESLI", "CRUESLI",
                           "CRUNCHY MUSLI", "ALPEN", "4-KORN", "SUPERGRØT",
                           "SUPERGRANOLA", "HAVREGRANOLA"]),
    ("havregryn_grøt",    ["HAVREGRYN", "HAVREGRØT", "HAVREMEL", "BJØRN HAVRE",
                           "SVARTHAVRERIS", "SVARTHAVREGRYN", "STEEL CUT"]),
    ("knekkebrød",        ["KNEKKEBRØD", "KNEKKS", "RUGSPRØ", "RYVITA", "KAVRINGER",
                           "WASA", "LEKSANDS", "SIGDAL", "KORNMO", "HUSMAN",
                           "FRUKOST KNEKKE", "HAVREKNEKKEBRØD", "HAVRE KNEKKEBRØD",
                           "DELIKATESS SESAM", "FIBER BALANCE", "DELICATE CRACKERS",
                           "POWERKNEKKEBRØD", "FRØKRISP"]),
    ("flatbrød_skorpor",  ["FLATBRØD", "SKORPOR", "MELBA TOAST", "MELBATOAST",
                           "BRUSCHETT", "GRISSINI", "SANDWICH WASA", "RISKAKER",
                           "MAISKAKER", "KRUTONGER", "SURDEIG"]),
    ("kjeks_kaker",       ["KJEKS", "GIFFLAR", "TOM&JERRY", "NUTELLA BISCUITS",
                           "SÆTRE", "FIBERRIK", "GULLBAR", "MELLOMBAR",
                           "DAVE&JON", "EXTRA SWEET"]),
    ("smør_margarin",     ["SMØR", "MARGARIN", "MEIERISMØR", "BREMYKT", "BRELETT",
                           "SMØREMYK", "SOFT FLORA", "VITA HJERTEGO", "OLIVERO",
                           "MELANGE", "FLOTT MATFETT", "STEKEMARGARIN",
                           "BORDMARGARIN", "FLORA"]),
    ("egg",               ["EGG", "GÅRDSEGG", "SOLEGG", "FRITTGÅENDE", "ØKOLOGISKE EGG",
                           "EGGEHVITE", "FROKOSTEGG"]),
    ("annet",             [""]),   # fallback — matcher alt
]


def merge_class_name(product_name: str) -> str:
    """Map a full product name to its merged category."""
    name_upper = product_name.upper()
    for category, keywords in CLASS_MERGE_RULES:
        for kw in keywords:
            if kw in name_upper:
                return category
    return "annet"


def build_merge_mapping(categories: dict[int, str]) -> tuple[dict[int, str], dict[int, int]]:
    """
    Returns:
        merged_names: {new_id: category_name}
        old_to_new:   {original_coco_id: new_yolo_id}
    """
    category_list = sorted(set(CLASS_MERGE_RULES[i][0] for i in range(len(CLASS_MERGE_RULES))))
    cat_to_id = {name: i for i, name in enumerate(category_list)}

    old_to_new = {}
    for coco_id, product_name in categories.items():
        merged = merge_class_name(product_name)
        old_to_new[coco_id] = cat_to_id[merged]

    merged_names = {v: k for k, v in cat_to_id.items()}
    return merged_names, old_to_new


def print_merge_summary(categories: dict[int, str], old_to_new: dict[int, int], merged_names: dict[int, str]):
    """Print how many original classes mapped to each category."""
    from collections import Counter
    counts = Counter(merged_names[new_id] for new_id in old_to_new.values())
    print("\n📦 Klassamerging:")
    print(f"  {'Kategori':<22} {'Antall originalklasser':>22}")
    print("  " + "─" * 46)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<22} {count:>22}")
    print(f"\n  356 klasser → {len(merged_names)} kategorier")


# ─────────────────────────────────────────────────────────────
#  COCO → YOLO conversion
# ─────────────────────────────────────────────────────────────

def coco_to_yolo(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    merge_classes: bool = True,
) -> dict[int, str]:
    """
    Convert COCO JSON annotations to per-image YOLO .txt files.
    Optionally merges 356 product names into ~15 categories.
    Returns {yolo_class_id: class_name}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_info = {img["id"]: img for img in coco["images"]}

    # Build class mapping
    if merge_classes:
        class_names, old_to_new = build_merge_mapping(categories)
        print_merge_summary(categories, old_to_new, class_names)
    else:
        # Keep all classes, remap to 0-indexed
        sorted_ids = sorted(categories)
        old_to_new = {cid: i for i, cid in enumerate(sorted_ids)}
        class_names = {i: categories[cid] for i, cid in enumerate(sorted_ids)}
        print(f"\n⚠️  Bruker alle {len(class_names)} klasser (ingen merging)")

    # Group annotations by image
    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Filter out empty-name and unknown categories
    skip_ids = {cid for cid, name in categories.items() if name.strip() == ""}

    print(f"\nKonverterer {len(image_info)} bilder...")
    skipped_box = 0
    skipped_class = 0

    for img_id, img in tqdm(image_info.items(), desc="COCO → YOLO"):
        W, H = img["width"], img["height"]
        label_lines = []

        for ann in ann_by_image.get(img_id, []):
            # Skip empty class names
            if ann["category_id"] in skip_ids:
                skipped_class += 1
                continue

            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            wn = w / W
            hn = h / H

            if wn <= 0 or hn <= 0:
                skipped_box += 1
                continue

            new_cls = old_to_new[ann["category_id"]]
            label_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        stem = Path(img["file_name"]).stem
        (output_dir / f"{stem}.txt").write_text("\n".join(label_lines))

    if skipped_box:
        print(f"⚠️  Hoppet over {skipped_box} ugyldige bounding boxes")
    if skipped_class:
        print(f"⚠️  Hoppet over {skipped_class} annoteringer med tomt klassenavn")

    return class_names


# ─────────────────────────────────────────────────────────────
#  Dataset splitting
# ─────────────────────────────────────────────────────────────

def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_exts])

    if not image_files:
        raise FileNotFoundError(f"Ingen bilder funnet i {images_dir}")

    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    splits = {
        "test":  image_files[:n_test],
        "val":   image_files[n_test:n_test + n_val],
        "train": image_files[n_test + n_val:],
    }

    print(f"\nDataset split ({n} bilder totalt):")
    for split, files in splits.items():
        print(f"  {split:5s}: {len(files):4d} bilder")

    for split, files in splits.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"Kopierer {split}"):
            shutil.copy2(img_path, img_out / img_path.name)
            label_src = labels_dir / (img_path.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, lbl_out / label_src.name)
            else:
                (lbl_out / (img_path.stem + ".txt")).touch()

    print("✅ Split ferdig\n")


# ─────────────────────────────────────────────────────────────
#  Write dataset YAML
# ─────────────────────────────────────────────────────────────

def write_dataset_yaml(output_dir: Path, class_names: dict[int, str]):
    dataset_yaml = {
        "path":  str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": [class_names[i] for i in range(len(class_names))],
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)
    print(f"📄 Dataset YAML: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────
#  Verification
# ─────────────────────────────────────────────────────────────

def verify_dataset(splits_dir: Path):
    print("\n🔍 Verifisering:")
    for split in ["train", "val", "test"]:
        img_dir = splits_dir / split / "images"
        lbl_dir = splits_dir / split / "labels"
        if not img_dir.exists():
            continue
        images = list(img_dir.iterdir())
        labels = list(lbl_dir.iterdir()) if lbl_dir.exists() else []
        empty  = sum(1 for l in labels if l.stat().st_size == 0)
        print(f"  {split:5s}: {len(images):4d} bilder, {len(labels):4d} labels ({empty} tomme)")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess grocery detection dataset")
    parser.add_argument("--data_dir",   type=str, default="res/dataset/train",
                        help="Mappe med images/ og annotations.json")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--format",     type=str, default="coco", choices=["coco", "yolo"])
    parser.add_argument("--no_merge",   action="store_true",
                        help="Behold alle 356 klasser (ikke anbefalt)")
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    labels_dir = data_dir / "labels_yolo"

    # Finn annotations.json — støtter flere mulige plasseringer
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        candidates = list((data_dir / "annotations").glob("*.json"))
        if candidates:
            ann_path = candidates[0]
            print(f"Bruker annotasjonsfil: {ann_path}")
        else:
            raise FileNotFoundError(
                f"Fant ikke annotations.json i {data_dir}.\n"
                f"Forventet: {data_dir}/annotations.json"
            )

    class_names = coco_to_yolo(
        annotations_path=ann_path,
        images_dir=data_dir / "images",
        output_dir=labels_dir,
        merge_classes=not args.no_merge,
    )

    split_dataset(
        images_dir=data_dir / "images",
        labels_dir=labels_dir,
        output_dir=output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    yaml_path = write_dataset_yaml(output_dir, class_names)
    verify_dataset(output_dir)

    print(f"\n✅ Preprocessing ferdig!")
    print(f"   Dataset YAML:  {yaml_path}")
    print(f"   Neste steg:    python3 src/train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()

