import numpy as np
import os
import pandas
import argparse
import time
import math
import h5py
import json
import torchvision.transforms as transforms
from PIL import Image
import configs

def main(args):
    verbose = args.verbose
    phases = args.phases
    if not phases:
        phases = ["train", "val", "test"]
    else:
        phases = [args.phases]
    coco_size = configs.COCO_SIZE
    coco_root = configs.COCO_ROOT
    for phase in phases:
        print("phase: ", phase)
        print()
        # settings
        print("creating database...")
        print()
        if phase == "train":
            coco_dir = os.path.join(coco_root, "%s2014" % "train")
        else:
            coco_dir = os.path.join(coco_root, "%s2014" % "val")
        coco_split = os.path.join(configs.COCO_ROOT, configs.COCO_SPLIT)
        coco_paths = None
        if not os.path.exists(os.path.join(coco_root, "preprocessed")):
            os.mkdir(os.path.join(coco_root, "preprocessed"))
        database = h5py.File(os.path.join(coco_root, "preprocessed", configs.COCO_DATABASE.format(phase, coco_size)), "w", libver='latest')  

        # processing captions
        print("creating preprocessed csv...")
        print()
        coco_split = json.load(open(coco_split))
        coco_split = [item for item in coco_split["images"] if item['split'] == phase]
        df_caption = {
            'image_id': [],
            'caption': []
        }
        for item in coco_split:
            for sub in item["sentences"]:
                df_caption["image_id"].append(sub["imgid"])
                df_caption["caption"].append(" ".join(sub["tokens"]))
        df_caption = pandas.DataFrame(df_caption, columns=['image_id', 'caption'])
        df_filename = {
            'image_id': [item['imgid'] for item in coco_split],
            'file_name': [item['filename'] for item in coco_split]
        }
        df_filename = pandas.DataFrame(df_filename, columns=['image_id', 'file_name'])
        coco_csv = df_caption.merge(df_filename, how="inner", left_on="image_id", right_on="image_id")
        coco_csv = coco_csv.sample(frac=1).reset_index(drop=True)
        # shuffle the dataset
        coco_csv.to_csv(os.path.join(coco_root, "preprocessed", configs.COCO_CAPTION.format(phase)), index=False)
        coco_paths = coco_csv.file_name.drop_duplicates().values.tolist()
        # create indices
        print("creating indices for images...")
        print()
        index = {coco_paths[i]: i for i in range(len(coco_paths))}
        __all = coco_csv.file_name.values.tolist()
        mapping = {i: index[__all[i]] for i in range(len(__all))}
        print("images:", len(index.keys()))
        print("pairs:", len(mapping.keys()))
        print("\nsaving indices...")
        print()
        json.dump(mapping, open(os.path.join(coco_root, "preprocessed", configs.COCO_INDEX.format(phase), "w")))


        # processing images
        print("preprocessing the images...")
        print()
        dataset = database.create_dataset("images", (len(coco_paths), 3 * coco_size * coco_size), dtype="float")
        trans = transforms.Compose([
            transforms.Resize(coco_size),
            transforms.CenterCrop(coco_size),
            transforms.ToTensor(),
        ])
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i, path in enumerate(coco_paths):
            start_since = time.time()
            image = Image.open(os.path.join(coco_dir, path))
            image = trans(image)
            if image.size(0) < 3:
                image = image.expand(3, image.size(1), image.size(2))
            else:
                image = image[:3]
            image = norm(image)
            image = image.contiguous().view(-1).numpy()
            dataset[i] = image
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(coco_paths) - i)
            eta_m = math.floor(eta_s / 60)
            if i % verbose == 0: 
                print("preprocessed and stored: %d, ETA: %dm %ds" % (i, eta_m, eta_s - eta_m * 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--phases", type=str, default=None, help="train/val/test")
    args = parser.parse_args()
    print(args)
    print()
    main(args)