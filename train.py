import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from encoders import *
from decoders import *
from solver import *
import matplotlib.pyplot as plt
import configs


def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    beam_size = args.beam_size
    epoch = args.epoch
    verbose = args.verbose
    lr = args.learning_rate
    batch_size = args.batch_size
    model_type = "coco"
    weight_decay = args.weight_decay
    if args.attention == "true":
        attention = True
    elif args.attention == "false":
        attention = False
    if args.evaluation == "true":
        evaluation = True
    elif args.evaluation == "false":
        evaluation = False
    pretrained = args.pretrained
    if pretrained:
        model_name = pretrained
    else:
        model_name = "shallow"

    print("\n[settings]")
    print("GPU:", args.gpu)
    print("pretrained:", args.pretrained)
    print("attention:", args.attention)
    print("evaluation:", args.evaluation)
    print("train_size:", args.train_size)
    print("val_size:", args.val_size)
    print("test_size:", args.test_size)
    print("beam_size:", args.beam_size)
    print("epoch:", args.epoch)
    print("verbose:", args.verbose)
    print("batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    print()

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                   training for encoder-decoder                  #
    #                                                                 #
    #                                                                 #
    ###################################################################

    # for coco
    # preprocessing
    print("preparing data....")
    print()
    coco = COCO(
        # for training
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("train"))), 
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("val"))),
        pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("test"))),
        [train_size, val_size, test_size]
        # # for debugging
        # pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("train"))), 
        # pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("train"))), 
        # pandas.read_csv(os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_CAPTION.format("train"))),
        # [train_size, val_size, test_size]
    )
    # split data
    train_captions = coco.transformed_data['train']
    val_captions = coco.transformed_data['val']
    test_captions = coco.transformed_data['test']
    dict_idx2word = coco.dict_idx2word
    dict_word2idx = coco.dict_word2idx
    corpus = coco.corpus
    # prepare the dataloader
    if pretrained:
        train_ds = COCOCaptionDataset(
            os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("train")), 
            train_captions, 
            database=os.path.join("data", configs.COCO_FEATURE.format("train", pretrained))
        )
        val_ds = COCOCaptionDataset(
            # for training
            os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("val")), 
            val_captions,
            database=os.path.join("data", configs.COCO_FEATURE.format("val", pretrained))
            # # for debugging
            # os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("train")), 
            # val_captions,
            # database=os.path.join("data", configs.COCO_FEATURE.format("train", pretrained))
        )
        test_ds = COCOCaptionDataset(
            # for training
            os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("test")), 
            test_captions, 
            database=os.path.join("data", configs.COCO_FEATURE.format("test", pretrained))
            # # for debugging
            # os.path.join(configs.COCO_ROOT, "preprocessed", configs.COCO_INDEX.format("test")), 
            # val_captions,
            # database=os.path.join("data", configs.COCO_FEATURE.format("train", pretrained))
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader = {
            'train': train_dl,
            'val': val_dl,
            'test': test_dl
        }
        # initialize the encoder
        if pretrained == "resnet":
            if attention:
                print("initializing encoder: resnet152 with attention....")
                print()
                encoder = AttentionEncoderResNet152().cuda()
            else:
                print("initializing encoder: resnet152....")
                print()
                encoder = EncoderResNet152().cuda()
        elif pretrained == "vgg":
            if attention:
                print("initializing encoder: vgg16_bn with attention....")
                print()
                encoder = AttentionEncoderVGG16BN().cuda()
            else:
                print("initializing encoder: vgg16_bn....")
                print()
                encoder = EncoderVGG16BN().cuda()
        else:
            print("inval model name, terminating...")
            return
    else:
        print("inval model type, terminating.....")
        return

    # define the decoder
    input_size = dict_word2idx.__len__()
    hidden_size = 512
    num_layer = 1
    if attention:
        if pretrained == "vgg":
            print("initializing decoder with attention....")
            decoder = AttentionDecoder2D(batch_size, input_size, hidden_size, 512, 14, num_layer).cuda()
        elif pretrained == "resnet":
            print("initializing decoder with attention....")
            decoder = AttentionDecoder2D(batch_size, input_size, hidden_size, 2048, 7, num_layer).cuda()
    else:
        print("initializing decoder without attention....")        
        decoder = Decoder(input_size, hidden_size, num_layer).cuda()
    print("input_size:", input_size)
    print("dict_size:", dict_word2idx.__len__())
    print("hidden_size:", hidden_size)
    print("num_layer:", num_layer)
    print()


    # prepare the training parameters
    if pretrained:
        if attention:
            params = list(decoder.parameters()) + list(encoder.global_mapping.parameters()) + list(encoder.area_mapping.parameters()) + list(encoder.area_bn.parameters())
        else:
            params = list(decoder.parameters()) + list(encoder.output_layer.parameters())
    else:
        params = list(decoder.parameters()) + list(encoder.conv_layer.parameters()) + list(encoder.fc_layer.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # training
    print("start training....")
    print()
    if attention:
        settings = "%s_%s_%s_trs%d_vs%d_ts%d_e%d_lr%f_wd%f_bs%d_vocal%d_beam%d" % (model_type, model_name, "attention", train_size, val_size, test_size, epoch, lr, weight_decay, batch_size, input_size, beam_size)
    else:
        settings = "%s_%s_%s_trs%d_vs%d_ts%d_e%d_lr%f_wd%f_bs%d_vocal%d_beam%d" % (model_type, model_name, "noattention", train_size, val_size, test_size, epoch, lr, weight_decay, batch_size, input_size, beam_size)
    encoder_decoder_solver = EncoderDecoderSolver(optimizer, criterion, model_type, settings)
    encoder, decoder = encoder_decoder_solver.train(encoder, decoder, dataloader, corpus, dict_word2idx, dict_idx2word, epoch, verbose, model_type, attention, beam_size)

    # save model
    print("saving the best models...\n")
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(encoder, "models/encoder_%s.pth" % settings)
    torch.save(decoder, "models/decoder_%s.pth" % settings)

    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    # val_losses = [encoder_decoder_solver.log[i]["val_loss"] for i in range(epochs)]train_perplexity
    train_blues_1 = [encoder_decoder_solver.log[i]["train_bleu_1"] for i in range(epochs)]
    train_blues_2 = [encoder_decoder_solver.log[i]["train_bleu_2"] for i in range(epochs)]
    train_blues_3 = [encoder_decoder_solver.log[i]["train_bleu_3"] for i in range(epochs)]
    train_blues_4 = [encoder_decoder_solver.log[i]["train_bleu_4"] for i in range(epochs)]
    val_blues_1 = [encoder_decoder_solver.log[i]["val_bleu_1"] for i in range(epochs)]
    val_blues_2 = [encoder_decoder_solver.log[i]["val_bleu_2"] for i in range(epochs)]
    val_blues_3 = [encoder_decoder_solver.log[i]["val_bleu_3"] for i in range(epochs)]
    val_blues_4 = [encoder_decoder_solver.log[i]["val_bleu_4"] for i in range(epochs)]
    train_cider = [encoder_decoder_solver.log[i]["train_cider"] for i in range(epochs)]
    val_cider = [encoder_decoder_solver.log[i]["val_cider"] for i in range(epochs)]
    # train_meteor = [encoder_decoder_solver.log[i]["train_meteor"] for i in range(epochs)]
    # val_meteor = [encoder_decoder_solver.log[i]["val_meteor"] for i in range(epochs)]
    train_rouge = [encoder_decoder_solver.log[i]["train_rouge"] for i in range(epochs)]
    val_rouge = [encoder_decoder_solver.log[i]["val_rouge"] for i in range(epochs)]

    # plot training curve
    print("plot training curves...")
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    # plt.plot(range(epochs), val_losses, label="val_loss")
    # plt.plot(range(epochs), train_perplexity, label="train_perplexity")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/training_curve_%s.png" % settings, bbox_inches="tight")
    # plot the bleu scores
    fig.clf()
    fig.set_size_inches(16,32)
    plt.subplot(4, 1, 1)
    plt.plot(range(epochs), train_blues_1, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_1, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-1')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(range(epochs), train_blues_2, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_2, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-2')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(range(epochs), train_blues_3, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_3, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-3')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(range(epochs), train_blues_4, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_4, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-4')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/bleu_curve_%s.png" % settings, bbox_inches="tight")
    # plot the cider scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_cider, label="train_cider")
    plt.plot(range(epochs), val_cider, label="val_cider")
    plt.xlabel('epoch')
    plt.ylabel('CIDEr')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/cider_curve_%s.png" % settings, bbox_inches="tight")
    # # plot the meteor scores
    # fig.clf()
    # fig.set_size_inches(16,8)
    # plt.plot(range(epochs), train_meteor, label="train_meteor")
    # plt.plot(range(epochs), val_meteor, label="val_meteor")
    # plt.xlabel('epoch')
    # plt.ylabel('METEOR')
    # plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    # plt.legend()
    # plt.savefig("figs/meteor_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # plot the rouge scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_rouge, label="train_rouge")
    plt.plot(range(epochs), val_rouge, label="val_rouge")
    plt.xlabel('epoch')
    plt.ylabel('ROUGE_L')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/rouge_curve_%s.png" % settings, bbox_inches="tight")



    ###################################################################
    #                                                                 #
    #                                                                 #
    #                             evaluation                          #
    #                                                                 #
    #                                                                 #
    ###################################################################
    if evaluation:
        print("\nevaluating with beam search...")
        print()
        encoder.eval()
        decoder.eval()
        beam_size = ['1', '3', '5', '7']
        candidates = {i:{} for i in beam_size}
        outputs = {i:{} for i in beam_size}
        bleu = {i:{} for i in beam_size}
        cider = {i:{} for i in beam_size}
        rouge = {i:{} for i in beam_size}
        if os.path.exists("scores/{}.json".format(settings)):
            print("loading existing results...")
            print()
            candidates = json.load(open("scores/{}.json".format(settings)))
        else:
            print("evaluating...")
            print()
            for _, (model_ids, visuals, captions, cap_lengths) in enumerate(dataloader['test']):
                visual_inputs = Variable(visuals, requires_grad=False).cuda()
                caption_inputs = Variable(captions[:, :-1], requires_grad=False).cuda()
                cap_lengths = Variable(cap_lengths, requires_grad=False).cuda()
                visual_contexts = encoder(visual_inputs)
                max_length = int(cap_lengths[0].item()) + 10
                for bs in beam_size:
                    if attention:
                        outputs[bs] = decoder.beam_search(visual_contexts, caption_inputs, bs, max_length)
                        outputs[bs] = decode_attention_outputs(outputs[bs], None, dict_idx2word, "val")
                    else:
                        outputs[bs] = decoder.beam_search(visual_contexts, bs, max_length)
                        outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
                    for model_id, output in zip(model_ids, outputs[bs]):
                        if model_id not in candidates[bs].keys():
                            candidates[bs][model_id] = [output]
                        else:
                            candidates[bs][model_id].append(output)
            # save results
            json.dump(candidates, open("scores/{}.json".format(settings), 'w'))

        for bs in beam_size:
            # compute
            bleu[bs] = capbleu.Bleu(4).compute_score(corpus['test'], candidates[bs])
            cider[bs] = capcider.Cider().compute_score(corpus['test'], candidates[bs])
            rouge[bs] = caprouge.Rouge().compute_score(corpus['test'], candidates[bs])
            # report
            print("----------------------Beam_size: {}-----------------------".format(bs))
            print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][0], max(bleu[bs][1][0]), min(bleu[bs][1][0])))
            print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][1], max(bleu[bs][1][1]), min(bleu[bs][1][1])))
            print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][2], max(bleu[bs][1][2]), min(bleu[bs][1][2])))
            print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][3], max(bleu[bs][1][3]), min(bleu[bs][1][3])))
            print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[bs][0], max(cider[bs][1]), min(cider[bs][1])))
            print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[bs][0], max(rouge[bs][1]), min(rouge[bs][1])))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--val_size", type=int, default=100, help="val size for input captions")
    parser.add_argument("--test_size", type=int, default=0, help="test size for input captions")
    parser.add_argument("--beam_size", type=int, default=1, help="beam size for online evaluation via beam search")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty on the optimizer")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--pretrained", type=str, default=None, help="vgg/resnet")
    parser.add_argument("--attention", type=str, default="false", help="true/false")
    parser.add_argument("--evaluation", type=str, default="false", help="true/false")
    args = parser.parse_args()
    main(args)
