import os
import json
from PIL import Image


# Configuration
LABELS_ROOT = os.path.expanduser("~/dataset-ninja/bdd100k_10k/bdd100k:-images-10k")

IMAGES_ROOT = os.path.expanduser("~/cs444-finalproject/Dataset")

ATTR_LABELS_ROOT = os.path.expanduser("~/cs444-finalproject/Labels/100k")

OUTPUT_DIR = os.path.expanduser("~/cs444-finalproject/Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = ["train", "val", "test"]

# ----------------------------------------
# Helper to build attribute lookup
# ----------------------------------------
def build_attr_lookup(label_dir):
    """
    Builds a dict:
        image_name.jpg -> {"weather": ..., "scene": ..., "timeofday": ...}
    from all JSON files in the 100K label directory.
    """
    lookup = {}
    if not os.path.exists(label_dir):
        return lookup

    json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
    for jf in json_files:
        path = os.path.join(label_dir, jf)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            name = data.get("name")
            if not name:
                continue
            attrs = data.get("attributes", {})
            out = {}
            for k in ("weather", "scene", "timeofday"):
                val = attrs.get(k)
                if val and val != "undefined":
                    out[k] = val
            lookup[name + ".jpg"] = out
        except Exception:
            continue
    return lookup



for split in splits:
    img_dir = os.path.join(IMAGES_ROOT, split)
    ann_dir = os.path.join(LABELS_ROOT, split, "ann")
    attr_dir = os.path.join(ATTR_LABELS_ROOT, split)
    output_path = os.path.join(OUTPUT_DIR, f"det_{split}_10k.json")

    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"⚠️ Skipping {split}: missing folders.")
        continue

    print(f"Processing {split} split...")
    print(f" → Reading attributes from {attr_dir}")

    attr_lookup = build_attr_lookup(attr_dir)
    print(f"   Found {len(attr_lookup)} attribute entries")

    image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    images, annotations, categories = [], [], {}
    img_id = ann_id = 0
    missing_labels = matched = 0

    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, f"{img_file}.json")

        if not os.path.exists(ann_path):
            missing_labels += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        attributes = attr_lookup.get(img_file, {})  # from 100K
        img_id += 1
        matched += 1

        images.append({
            "id": img_id,
            "file_name": img_file,
            "height": h,
            "width": w,
            "attributes": attributes
        })

        # Read DatasetNinja annotation for bounding boxes
        with open(ann_path, "r") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            cat = obj.get("classTitle", "unknown")
            pts = obj.get("points", {}).get("exterior", [])
            if len(pts) < 2:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x, y, w_box, h_box = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

            if cat not in categories:
                categories[cat] = len(categories) + 1

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[cat],
                "bbox": [x, y, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0
            })

    # Build COCO-style dictionary
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved {split} to {output_path}")
    print(f"   Images processed: {len(images)}")
    print(f"   Matched labels: {matched}")
    print(f"   Missing labels: {missing_labels}")
    print(f"   Total annotations: {len(annotations)}")
    print(f"   Total categories: {len(categories)}")
