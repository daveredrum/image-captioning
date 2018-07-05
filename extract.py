import h5py
import pandas as pd
import numpy as np
import torch
import encoders
import data
import time
import math
import os
import pickle
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import configs

def main(args):
    coco_size = configs.COCO_SIZE
    if not args.phases:
        phases = ["train", "val", "test"]
    else:
        phases = [args.phases]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("\ninitializing model...")
    print()
    if args.pretrained == "vgg":
        model = encoders.VGG16BN().cuda()
    elif args.pretrained == "resnet":
        model = encoders.ResNet152().cuda()
    for phase in phases:
        print(phase)
        print()
        print("preparing...")
        print()
        dataset = data.FeatureDataset(
            database=os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_DATABASE.format(phase, coco_size))
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        if not os.path.exists("data/"):
            os.mkdir("data/")
        feature_path = os.path.join("data", configs.COCO_EXTRACTED.format(phase, args.pretrained))
        preprocessed_csv = pd.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format(phase)))
        
        # creat index
        # from preprocessed to raw
        __all = preprocessed_csv.image_id.values.tolist()
        image_ids = preprocessed_csv.image_id.drop_duplicates().values.tolist()
        index = {image_ids[i]: i for i in range(len(image_ids))}
        mapping = {i: index[__all[i]] for i in range(len(__all))}
        offset = 0
        print("extracting...")
        print()
        storage = []
        for images in dataloader:
            start_since = time.time()
            images = Variable(images).cuda()
            features = model(images)
            batch_size = features.size(0)
            for idx in range(batch_size):
                storage.append(features[idx].view(-1).data.cpu().numpy())
            offset += batch_size
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(dataset) - offset)
            eta_m = math.floor(eta_s / 60)
            print("preprocessed: %d, ETA: %dm %ds" % (offset, eta_m, eta_s - eta_m * 60))

        print("\nprocessing...\n")
        extracted = {}
        for i, item in enumerate(preprocessed_csv.values.tolist()):
            image_id = str(item[0])
            if image_id in extracted.keys():
                extracted[image_id].append(
                    [
                        item[1],
                        item[2],
                        storage[mapping[i]]
                    ]
                )
            else:
                extracted[image_id] = [
                    [
                        item[1],
                        item[2],
                        storage[mapping[i]]
                    ]
                ]
        
        print("storing..\n")
        pickle.dump(extracted, open(feature_path, 'wb'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, help="vgg/resnet")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--phases", type=str, default=None, help="train/val/test")
    args = parser.parse_args()
    main(args)
