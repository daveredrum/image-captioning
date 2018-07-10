import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from data import *
from decoders import *
from solver import *
import matplotlib
import matplotlib.pyplot as plt
import configs

def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    phase = args.phase
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    pretrained = args.pretrained
    num = args.num
    encoder_path = "outputs/models/{}".format(args.encoder)
    decoder_path = "outputs/models/{}".format(args.decoder)
    dir_name = args.encoder[8:-4]
    coco_captions = pandas.read_csv(os.path.join(configs.COCO_ROOT, 'preprocessed', configs.COCO_CAPTION.format(phase)))
    print("\n[settings]")
    print("phase:", phase)
    print("num:", num)
    print("dir_name:", dir_name)
    print("encoder_path:", encoder_path)
    print("decoder_path:", decoder_path)
    print()
    
    # preprocessing
    print("preparing data....")
    print()
    coco = COCO(
        [
            pickle.load(open(os.path.join("data", configs.COCO_EXTRACTED.format("train", pretrained)), 'rb')),
            pickle.load(open(os.path.join("data", configs.COCO_EXTRACTED.format("val", pretrained)), 'rb')),
            pickle.load(open(os.path.join("data", configs.COCO_EXTRACTED.format("test", pretrained)), 'rb'))
        ],
        [
            train_size, 
            val_size, 
            test_size
        ]
    )
    # split data
    transformed = coco.transformed_data[phase]
    dict_word2idx = coco.dict_word2idx
    dict_idx2word = coco.dict_idx2word
    dataset = COCOCaptionDataset(transformed)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # testing
    print("initializing encoder and decoder...")
    print()
    encoder_decoder = AttentionEncoderDecoder(encoder_path, decoder_path, pretrained)

    if not os.path.exists("outputs/vis/{}".format(dir_name)):
            os.mkdir("outputs/vis/{}".format(dir_name))
    for idx in range(num):
        # testing
        print("testing :", idx)
        print("generating captions...")
        for i, (model_id, visual_inputs, _, _) in enumerate(dataloader):
            if i == idx:
                visual_inputs = Variable(visual_inputs).cuda()
                descriptions = " ".join(encoder_decoder.generate_text(visual_inputs, dict_word2idx, dict_idx2word, 20))
                pairs = encoder_decoder.visual_attention(visual_inputs, dict_word2idx, dict_idx2word, 20)
                break

         # plot testing results
        outname = "{}_{}".format(dir_name, idx)
        if not os.path.exists("outputs/vis/{}/{}".format(dir_name, outname)):
            os.mkdir("outputs/vis/{}/{}".format(dir_name, outname))
        print("saving results...")
        df = {"step {}".format(i+1): pairs[i][1].view(-1).data.cpu().numpy().tolist() for i in range(len(pairs))}
        df = pandas.DataFrame(df, columns=['step {}'.format(str(i+1)) for i in range(len(pairs))])
        df.to_csv("outputs/vis/{}/{}/{}.csv".format(dir_name, outname, outname), index=False)

        # subplot settings
        num_col = 4
        num_row = len(pairs) // num_col + 2
        subplot_size = 4

        # graph settings
        plt.switch_backend("agg")
        fig = plt.figure(dpi=100)
        fig.set_size_inches(subplot_size * num_col, subplot_size * (num_row + 1))
        fig.set_facecolor('white')
        
        # generate caption results
        print("visualizing results...")
        plt.subplot2grid((num_row, num_col), (0, 0))
        image_path = coco_captions.file_name.loc[coco_captions.image_id == int(model_id[0])].drop_duplicates().iloc[0]
        if phase == 'val' or phase == 'test':
            image = Image.open(os.path.join(configs.COCO_ROOT, "{}2014".format('val'), image_path)).resize((configs.COCO_SIZE, configs.COCO_SIZE))
        else:
            image = Image.open(os.path.join(configs.COCO_ROOT, "{}2014".format('train'), image_path)).resize((configs.COCO_SIZE, configs.COCO_SIZE))
        plt.imshow(image)
        plt.axis('off')
        plt.subplot2grid((num_row, num_col), (0, 1), colspan=num_col-1)
        plt.text(0, 0.5, descriptions, fontsize=16)
        plt.axis('off')
        
        # visualize attention weights
        print("visualizing attention weights...\n")
        for i in range(len(pairs)):
            plt.subplot2grid((num_row, num_col), (i // num_col + 1, i % num_col))
            plt.imshow(pairs[i][2].data.cpu().numpy())
            plt.text(0, 0, pairs[i][0], fontsize=16, color='black', backgroundcolor='white')
            plt.axis('off')


        # fig.tight_layout()
        plt.savefig("outputs/vis/{}/{}/{}.png".format(dir_name, outname, outname), bbox_inches="tight")
        fig.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, help="train/val/test")
    parser.add_argument("--pretrained", type=str, help="vgg/resnet")
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--num", type=int, default=0, help="number of the testing image")
    parser.add_argument("--encoder", type=str, default=None, help="path to the encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to the decoder")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    args = parser.parse_args()
    main(args)
