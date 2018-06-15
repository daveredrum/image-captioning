import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
import torch
import torch.nn as nn
from data import *
from encoders import *
from decoders import *
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
import capeval.meteor.meteor as capmeteor
import capeval.rouge.rouge as caprouge
from utils import *
import configs
import matplotlib.pyplot as plt


class Report():
    def __init__(self, corpus, candidates, cider, num=3):
        '''
        params:
        - corpus: a dict containing image_ids and corresponding captions
        - candidates: a dict containing several dicts indexed by different beam sizes, in which images_ids
                      and corresponding generated captions are stored
        - cider: a dict containing several tuples indexed by different beam sizes, in which the mean CIDEr
                      scores and the per-image CIDEr scores are stored
        - num: number of images shown in the report, 3 by default
        '''
        self.corpus = corpus
        self.candidates = candidates
        self.cider = cider
        self.num = num
        self.beam_sizes = list(candidates.keys())
        self.image_ids = list(corpus.keys())
        self.chosen = self._pick()
        self.csv = pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("test")))

    def _pick(self):
        # return a dict of dicts containing images and captions
        chosen = {bs:None for bs in self.beam_sizes}
        for bs in self.beam_sizes:
            assert set(self.image_ids) == set(self.candidates[bs].keys())
            pairs = [(x, y) for x, y in zip(self.image_ids, self.cider[bs][1])]
            # choose the images with the highest scores, picking the first caption in candidates
            highest = sorted(pairs, reverse=True, key=lambda x: x[1])[:self.num]
            highest = [(highest[i][0], highest[i][1], self.candidates[bs][highest[i][0]][0], self.corpus[highest[i][0]][0]) for i in range(len(highest))]
            # the same thing for the lowest
            lowest = sorted(pairs, key=lambda x: x[1])[:self.num]
            lowest = [(lowest[i][0], lowest[i][1], self.candidates[bs][lowest[i][0]][0], self.corpus[lowest[i][0]][0]) for i in range(len(lowest))]
            # choose the images with the closest scores to the mean scores
            med_pairs = [(x, abs(y - self.cider[bs][0])) for x, y in zip(self.image_ids, self.cider[bs][1])]
            med = sorted(med_pairs, key=lambda x: x[1])[:self.num]
            med = [(med[i][0], med[i][1], self.candidates[bs][med[i][0]][0], self.corpus[med[i][0]][0]) for i in range(len(med))]
            # add into chosen
            chosen[bs] = {
                'high': highest,
                'low': lowest,
                'medium': med
            }
        
        return chosen
    
    def __call__(self, path):
        for q in ["high", "low", "medium"]:
            fig = plt.figure(dpi=100)
            fig.set_size_inches(8, 24)
            fig.set_facecolor('white')
            fig.clf()
            for i in range(self.num):
                image_id = int(self.chosen['1'][q][i][0])
                plt.subplot(self.num, 1, i+1)
                plt.imshow(Image.open(os.path.join(configs.COCO_ROOT, "val2014/{}".format(self.csv.loc[self.csv.image_id == image_id].file_name.drop_duplicates().iloc[0]))).convert('RGBA').resize((224, 224)))
                plt.text(240, 60, 'beam size 1 : ' + self.chosen['1'][q][i][2], fontsize=28)
                plt.text(240, 90, 'beam size 3 : ' + self.chosen['3'][q][i][2], fontsize=28)
                plt.text(240, 120, 'beam size 5 : ' + self.chosen['5'][q][i][2], fontsize=28)
                plt.text(240, 150, 'beam size 7 : ' + self.chosen['7'][q][i][2], fontsize=28)
                plt.text(240, 180, 'ground truth : ' + self.chosen['7'][q][i][3], fontsize=28)
                plt.axis('off')
            if os.path.exists("results/{}".format(path)):
                os.mkdir("results/{}".format(path))
            plt.savefig("results/{}/{}.png".format(path, q), bbox_inches="tight")


###################################################################
#                                                                 #
#                                                                 #
#                             evaluation                          #
#                                                                 #
#                                                                 #
###################################################################

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_size = args.train_size
    test_size = args.test_size
    num = args.num
    batch_size = args.batch_size
    pretrained = args.pretrained
    encoder_path = args.encoder
    decoder_path = args.decoder
    if args.attention == "true":
        attention = True
    elif args.attention == "false":
        attention = False
    outname = encoder_path.split('.')[8:]
    print("\n[settings]")
    print("GPU:", args.gpu)
    print("train_size:", args.train_size)
    print("test_size:", args.test_size)
    print("batch_size:", args.batch_size)
    print("pretrained:", args.pretrained)
    print("outname:", outname)
    print()
    print("preparing data...")
    print()
    coco = COCO(
        # for training
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("train"))), 
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("val"))),
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("test"))),
        [train_size, 0, test_size]
    )
    dict_idx2word = coco.dict_idx2word
    corpus = coco.corpus["test"]
    test_ds = COCOCaptionDataset(
        os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("test")), 
            coco.transformed_data['test'], 
            database=os.path.join("data", configs.COCO_FEATURE.format("test", pretrained))
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    print("initializing models...")
    print()
    encoder = torch.load(os.path.join("models", encoder_path)).cuda()
    decoder = torch.load(os.path.join("models", decoder_path)).cuda()
    encoder.eval()
    decoder.eval()
    beam_size = ['1', '3', '5', '7']
    candidates = {i:{} for i in beam_size}
    outputs = {i:{} for i in beam_size}
    bleu = {i:{} for i in beam_size}
    cider = {i:{} for i in beam_size}
    rouge = {i:{} for i in beam_size}
    if os.path.exists("scores/{}.json".format(outname)):
        print("loading existing results...")
        print()
        candidates = json.load(open("scores/{}.json".format(outname)))
    else:
        print("\nevaluating with beam search...")
        print()
        for _, (model_ids, visuals, captions, cap_lengths) in enumerate(test_dl):
            visual_inputs = Variable(visuals, requires_grad=False).cuda()
            caption_inputs = Variable(captions[:, :-1], requires_grad=False).cuda()
            cap_lengths = Variable(cap_lengths, requires_grad=False).cuda()
            visual_contexts = encoder(visual_inputs)
            max_length = int(cap_lengths[0].item()) + 10
            for bs in beam_size:
                if attention:
                    outputs[bs] = decoder.beam_search(visual_contexts, caption_inputs, int(bs), max_length)
                    outputs[bs] = decode_attention_outputs(outputs[bs], None, dict_idx2word, "val")
                else:
                    outputs[bs] = decoder.beam_search(visual_contexts, int(bs), max_length)
                    outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
                for model_id, output in zip(model_ids, outputs[bs]):
                    if model_id not in candidates[bs].keys():
                        candidates[bs][model_id] = [output]
                    else:
                        candidates[bs][model_id].append(output)
        # save results
        json.dump(candidates, open("scores/{}.json".format(outname), 'w'))

    for bs in beam_size:
        # compute
        bleu[bs] = capbleu.Bleu(4).compute_score(corpus, candidates[bs])
        cider[bs] = capcider.Cider().compute_score(corpus, candidates[bs])
        rouge[bs] = caprouge.Rouge().compute_score(corpus, candidates[bs])
        # report
        print("----------------------Beam_size: {}-----------------------".format(bs))
        print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][0], max(bleu[bs][1][0]), min(bleu[bs][1][0])))
        print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][1], max(bleu[bs][1][1]), min(bleu[bs][1][1])))
        print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][2], max(bleu[bs][1][2]), min(bleu[bs][1][2])))
        print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][3], max(bleu[bs][1][3]), min(bleu[bs][1][3])))
        print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[bs][0], max(cider[bs][1]), min(cider[bs][1])))
        print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[bs][0], max(rouge[bs][1]), min(rouge[bs][1])))
        print()
    
    # save figs
    report = Report(corpus, candidates, cider, num)
    report(outname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--test_size", type=int, default=0, help="test size for input captions")
    parser.add_argument("--num", type=int, default=3, help="number of shown images")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--pretrained", type=str, default=None, help="vgg/resnet")
    parser.add_argument("--attention", type=str, default=None, help="true/false")
    parser.add_argument("--encoder", type=str, default=None, help="path to encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to decoder")
    args = parser.parse_args()
    main(args)