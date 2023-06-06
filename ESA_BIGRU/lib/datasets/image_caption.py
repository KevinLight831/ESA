"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2
import nltk

import logging

logger = logging.getLogger(__name__)

class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, vocab, opt, train):
        self.vocab = vocab
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]

        if self.train:
            # Convert caption (string) to word ids (with Size Augmentation at training time).
            image = self.images[img_index]
			# Size augmentation on region features.
            hardnum = self.opt.hardnum
            num_features = image.shape[0]
            target = []
            target_len = [] 
            for i in range(hardnum):
                target_i = process_caption(self.vocab, caption,self.train,0.2)
                target_len.append(len(target_i))
                target.append(target_i)
            img_len = len(image)

            images = torch.Tensor(image)
            return images, target, img_len, target_len, index, img_index
        else:
            target = process_caption(self.vocab, caption)
            images = self.images[img_index]
            images = torch.Tensor(images)
            return images, target, index, img_index       

    def __len__(self):
        return self.length


def process_caption(vocab, caption, drop=False, prob_i=0.2):
    if not drop:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target
    else:
        # Convert caption (string) to word ids.
        tokens = ['<start>', ]
        tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokens.append('<end>')
        deleted_idx = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < prob_i:
                # tokens[i] = vocab(token)
                # deleted_idx.append(i)
                prob /= prob_i
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)
        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
        target = torch.Tensor(tokens)
        return target

def collate_fn_test(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions,  ids, img_ids = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)
        # Merget captions
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, ids

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, img_lens, cap_lens, ids, img_ids = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)
        # Merget captions
        cap_len = torch.Tensor(cap_lens).long()
        all_captions = torch.zeros(len(captions),len(cap_len[0]),cap_len.max()).long()
        for i,cap in enumerate(captions):
            for j,index in enumerate(cap_lens[i]):
                end = index
                all_captions[i,j,:end] = cap[j][:end]
        cap_lens = torch.Tensor(cap_lens)
        return all_images, img_lengths, all_captions, cap_lens, ids



def get_loader(data_path, data_name, data_split, vocab, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn,
                                                num_workers=num_workers,
                                                drop_last=True)
    else:
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_test,
                                                num_workers=num_workers,
                                                drop_last=False)
        
    return data_loader


def get_loaders(data_path, data_name, vocab, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', vocab, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, vocab, opt,
                             batch_size, False, workers, train=False)
    return test_loader
