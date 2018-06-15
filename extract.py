import h5py
import pandas as pd
import numpy as np
import torch
import encoders
import data
import time
import math
import os
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
        database = h5py.File(os.path.join("data", configs.COCO_FEATURE.format(phase, args.pretrained)), "w", libver='latest')
        if args.pretrained == "vgg":
            storage = database.create_dataset("features", (len(dataset), 512 * 14 * 14), dtype="float")
        elif args.pretrained == "resnet":
            storage = database.create_dataset("features", (len(dataset), 2048 * 7 * 7), dtype="float")
        offset = 0
        print("extracting...")
        print()
        for images in dataloader:
            start_since = time.time()
            images = Variable(images).cuda()
            features = model(images)
            batch_size = features.size(0)
            for idx in range(batch_size):
                storage[offset + idx] = features[idx].view(-1).data.cpu().numpy()
            offset += batch_size
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(dataset) - offset)
            eta_m = math.floor(eta_s / 60)
            print("preprocessed and stored: %d, ETA: %dm %ds" % (offset, eta_m, eta_s - eta_m * 60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, help="vgg/resnet")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--phases", type=str, default=None, help="train/val/test")
    args = parser.parse_args()
    main(args)
