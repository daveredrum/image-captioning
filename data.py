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
    def __init__(self, index_path, csv_file, database):
        self.index = json.load(open(index_path, "r"))
        self.model_ids = copy.deepcopy(csv_file.image_id.values.tolist())
        self.caption_lists = copy.deepcopy(csv_file.caption.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.database = h5py.File(database, "r")

    def __len__(self):
        return self.csv_file.image_id.count()

    def __getitem__(self, idx):
        # return (model_id, image_inputs, padded_caption, cap_length)
        image = self.database["features"][self.index[str(self.csv_file.iloc[idx].index)]]
        image = torch.FloatTensor(image)

        return str(self.model_ids[idx]), image, self.caption_lists[idx], len(self.caption_lists[idx])

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
    def __init__(self, train_csv, val_csv, test_csv, size_split):
        # size settings
        self.total_size = np.sum(size_split)
        self.train_size, self.val_size, self.test_size = size_split
        # select data by the given total size
        self.original_csv = {
            'train': None,
            'val': None,
            'test': None
        }
        # use image_id to select data
        # training set
        if self.train_size == -1:
            self.original_csv['train'] = train_csv
        else:
            self.original_csv['train'] = train_csv.iloc[:self.train_size]
        # valation set
        if self.val_size == -1:
            self.original_csv['val'] = val_csv
        else:
            self.original_csv['val'] = val_csv.iloc[:self.val_size]
        # testing set
        if self.test_size == -1:
            self.original_csv['test'] = test_csv
        else:
            self.original_csv['test'] = test_csv.iloc[:self.test_size]
        # dictionaries
        self.dict_word2idx = None
        self.dict_idx2word = None
        self.dict_size = None
        # ground truth captions grouped by image_id
        # for calculating BLEU score
        self.corpus = {
            'train': {},
            'val': {},
            'test': {}
        }
        # split the preprocessed data
        self.preprocessed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        # split the transformed data
        self.transformed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        
        # preprcess and transform
        self._preprocess()
        self._tranform()
        # build reference corpus
        self._make_corpus()

    # output the dictionary of captions
    # indices of words are the rank of frequencies
    def _make_dict(self):
        captions_list = self.preprocessed_data["train"].caption.values.tolist()
        # captions_list += self.preprocessed_data["val"].caption.values.tolist() 
        # captions_list += self.preprocessed_data["test"].caption.values.tolist()
        word_list = {}
        for text in captions_list:
            try:
                for word in re.split("[ ]", text):
                    if word and word != "<START>" and word != "<END>":
                        # set the maximum size of vocabulary
                        if word in word_list.keys():
                            word_list[word] += 1
                        else:
                            word_list[word] = 1
            except Exception:
                pass
        # max dict_size = 10000
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)[:10000]
        # indexing starts at 1
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
            for _, item in self.preprocessed_data[phase].iterrows():
                if str(item.image_id) in self.corpus[phase].keys():
                    self.corpus[phase][str(item.image_id)].append(item.caption)
                else:
                    self.corpus[phase][str(item.image_id)] = [item.caption]


    def _preprocess(self):
        # suppress all warnings
        warnings.simplefilter('ignore')
        for phase in ["train", "val", "test"]:
            # drop items without captions
            self.preprocessed_data[phase] = copy.deepcopy(self.original_csv[phase].loc[self.original_csv[phase].caption.notnull()].reset_index(drop=True))
            # convert to lowercase
            self.preprocessed_data[phase].caption = self.preprocessed_data[phase].caption.str.lower()
            # preprocess
            captions_list = self.preprocessed_data[phase].caption.values.tolist()
            for i in range(len(captions_list)):
                caption = captions_list[i]
                # truncate long captions
                max_length = 18
                caption = caption.split(" ")
                if len(caption) > max_length:
                    caption = caption[:max_length]
                caption = " ".join(caption)
                # add start symbol
                caption = '<START> ' + caption
                # add end symbol
                caption += ' <END>'
                captions_list[i] = caption
                # filter out empty element
                caption = filter(None, caption)
            # replace with the new column
            new_captions = pandas.DataFrame({'caption': captions_list})
            self.preprocessed_data[phase].caption = new_captions.caption

        # build dict
        self._make_dict()

    # transform all words to their indices in the dictionary
    def _tranform(self):
        for phase in ["train", "val", "test"]:
            if phase == "val" or phase == "test":
                self.transformed_data[phase] = copy.deepcopy(self.preprocessed_data[phase]).drop_duplicates(subset='image_id')
            else:
                self.transformed_data[phase] = copy.deepcopy(self.preprocessed_data[phase])
            captions_list = self.transformed_data[phase].caption.values.tolist()
            for i in range(len(captions_list)):
                temp_list = []
                for text in captions_list[i].split(" "):
                    # filter out empty element
                    if text and text in self.dict_word2idx.keys():
                        temp_list.append(self.dict_word2idx[text])
                    elif text and text not in self.dict_word2idx.keys():
                        temp_list.append(self.dict_word2idx["<UNK>"])
                    captions_list[i] = temp_list
            # replace with the new column
            transformed_captions = pandas.DataFrame({'caption': captions_list})
            self.transformed_data[phase].caption = transformed_captions.caption
