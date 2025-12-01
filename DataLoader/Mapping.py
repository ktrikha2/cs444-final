import json, os
from PIL import Image 

## IO Locations
Images = r"Dataset"
Labels = r"/home/ktrikha/cs444-finalproject/Labels/100k"
Output = r"cs444-final-project/data"

## COCO format for test, train, val
splits = ["train", "val", "test"]

for split in splits:
    img_dir = os.path.join(Images, split)
    label_dir = os.path.join(Labels, split)
    output_json = os.path.join(Output, f"det_{split}_10k.json")

    json_files = sorted(os.listdir(label_dir))
    #print("label_file",label_files[1])

    images = []
    annotations = []
    categories = {}
    img_id = 0
    ann_id = 0

    for json_file in json_files:

        ## Load each json file  
        i_json_path = os.path.join(label_dir, json_file)
        with open(i_json_path, 'r') as f:
            data = json.load(f)

        ## Check if the image actually exists in the 10k 
        img_name = data["name"] + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[{split}], image:{img_name} -> image doesnt exist")
            continue

        ## Add image size (not hard coded) 
        with Image.open(img_path) as im:
            width, height = im.size

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": img_name,
            "height": height,
            "width": width,
            "attributes": data.get("attributes", {})

        })

        ## Add object details 

        objs = []
        if "frames" in data:
            frames = data.get("frames", [])
            if frames:
                objs = frames[0].get("objects", [])

        elif "labels" in data:
            objs = data.get("labels", [])

        print(objs[0])

        for obj in objs:
            cat_name = obj.get("category")
            if not cat_name:
                continue

            # box2d for object detection
            if "box2d" not in obj:
                continue

            if cat_name not in categories:
                categories[cat_name] = len(categories) + 1

            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            width_box, height_box = x2 - x1, y2 - y1

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[cat_name],
                "bbox": [x1, y1, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0,
                "attributes": obj.get("attributes", {}) 
            })

    coco_categories = [{"id": cid, "name": name} for name, cid in categories.items()]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"[{split}] COCO JSON saved: {output_json}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}, Categories: {len(coco_categories)}\n")



