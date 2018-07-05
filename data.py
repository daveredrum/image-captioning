import torch
import pandas
import os
import re
import warnings
import operator
import copy
import nrrd
import math
import h5py
import pickle
import random
import string
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# dataset for coco
class COCOCaptionDataset(Dataset):
    def __init__(self, transformed_data):
        self.transformed_data = transformed_data

    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        # return (model_id, feature, caption, cap_length)
        model_id = self.transformed_data[idx][0]
        caption = self.transformed_data[idx][1]
        feature = self.transformed_data[idx][2]
        feature = torch.FloatTensor(feature)

        return model_id, feature, caption, len(caption)

def collate_fn(data):
    '''
    new method to merge data into mini-batches
    callable while initializing dataloader by collate_fn=collate_fn
    '''
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: x[3], reverse=True)
    model_ids, images, captions, lengths = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])


    return model_ids, images, targets, torch.Tensor([l for l in lengths])

class FeatureDataset(Dataset):
    def __init__(self, database):
        self.database = h5py.File(database, "r")

    def __len__(self):
        return self.database["images"].shape[0]

    def __getitem__(self, idx):
        image = self.database["images"][idx]
        image = torch.FloatTensor(image).view(3, 224, 224)

        return image


# process coco csv file
class COCO(object):
    def __init__(self, data_split, size_split):
        # size settings
        self.total_size = np.sum(size_split)
        self.train_size, self.val_size, self.test_size = size_split
        
        # sets
        train_data, val_data, test_data = data_split
        self.original_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # training set
        if self.train_size != -1:
            train_data = [(item[0], item[1]) for item in train_data.items()][:self.train_size]
            self.original_data['train'] = {item[0]: item[1] for item in train_data}
       
        # valation set
        if self.val_size != -1:
            val_data = [(item[0], item[1]) for item in val_data.items()][:self.val_size]
            self.original_data['val'] = {item[0]: item[1] for item in val_data}
        
        # testing set
        if self.test_size != -1:
            test_data = [(item[0], item[1]) for item in test_data.items()][:self.test_size]
            self.original_data['test'] = {item[0]: item[1] for item in test_data}
        
        # dictionaries
        self.dict_word2idx = {}
        self.dict_idx2word = {}
        self.dict_size = {}
        # ground truth captions grouped by image_id
        # for calculating BLEU score
        self.corpus = {
            'train': {},
            'val': {},
            'test': {}
        }
        # split the preprocessed data
        self.preprocessed_data = {
            'train': {},
            'val': {},
            'test': {}
        }
        # split the transformed data
        self.transformed_data = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        # preprcess and transform
        self._preprocess()
        self._tranform()
        # build reference corpus
        self._make_corpus()

    # output the dictionary of captions
    # indices of words are the rank of frequencies
    def _make_dict(self):
        word_list = {}
        for image_id in self.preprocessed_data['train'].keys():
            for item in self.preprocessed_data['train'][image_id]:
                for word in item[0].split(' '):
                    if word and word != "<START>" and word != "<END>":
                        if word in word_list.keys():
                            word_list[word] += 1
                        else:
                            word_list[word] = 1
        
        # max dict_size = 10000
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)[:10000]
        # indexing starts at 4
        self.dict_word2idx = {word_list[i][0]: i+4 for i in range(len(word_list))}
        self.dict_idx2word = {i+4: word_list[i][0] for i in range(len(word_list))}
        # add special tokens
        self.dict_word2idx["<PAD>"] = 0
        self.dict_idx2word[0] = "<PAD>"
        self.dict_word2idx["<UNK>"] = 1
        self.dict_idx2word[1] = "<UNK>"
        self.dict_word2idx["<START>"] = 2
        self.dict_idx2word[2] = "<START>"
        self.dict_word2idx["<END>"] = 3
        self.dict_idx2word[3] = "<END>"
        # dictionary size
        assert self.dict_idx2word.__len__() == self.dict_word2idx.__len__()
        self.dict_size = self.dict_idx2word.__len__()        

        
    # build the references for calculating BLEU score
    # return the dictionary of image_id and corresponding captions
    # input must be the preprocessed csv！
    def _make_corpus(self):
        for phase in ["train", "val", "test"]:
            for image_id in self.preprocessed_data[phase].keys():
                for item in self.preprocessed_data[phase][image_id]:
                    if image_id in self.corpus[phase].keys():
                        self.corpus[phase][image_id].append(item[0])
                    else:
                        self.corpus[phase][image_id] = [item[0]]

    def _preprocess(self):
        for phase in ["train", "val", "test"]:
            preprocessed_data = {}
            for image_id in self.original_data[phase].keys():
                preprocessed_data[image_id] = []
                for item in self.original_data[phase][image_id]:
                    caption = item[0]
                    # convert to lowercase
                    caption = caption.lower()
                    # truncate long captions
                    max_length = 18
                    caption = caption.split(" ")
                    if len(caption) > max_length:
                        caption = caption[:max_length]
                    caption = ' '.join(caption)
                    # add start symbol
                    caption = '<START> ' + caption
                    # add end symbol
                    caption += ' <END>'
                    
                    # store
                    preprocessed_data[image_id].append(
                        [
                            caption,
                            item[1],
                            item[2]
                        ]
                    )
            
            # store to objective
            self.preprocessed_data[phase] = preprocessed_data

        # build dict
        self._make_dict()

    # transform all words to their indices in the dictionary
    def _tranform(self):
        for phase in ["train", "val", "test"]:
            transformed_data = []
            for image_id in self.preprocessed_data[phase].keys():
                for item in self.preprocessed_data[phase][image_id]:
                    caption = []
                    for word in item[0].split(" "):
                        # filter out empty element
                        if word and word in self.dict_word2idx.keys():
                            caption.append(self.dict_word2idx[word])
                        elif word and word not in self.dict_word2idx.keys():
                            caption.append(self.dict_word2idx["<UNK>"])
                    
                    # store
                    transformed_data.append(
                        [
                            image_id,
                            caption,
                            item[1],
                            item[2]
                        ]
                    )
                    
                    # only choose one sample for val/test
                    if phase == 'val' or phase == 'test':
                        break
            
            # store to objective
            self.transformed_data[phase] = transformed_data 
