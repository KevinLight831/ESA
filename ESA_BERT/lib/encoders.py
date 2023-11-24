"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.init_weights()


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image):
        """Extract image feature vectors."""

        features = self.fc(image)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for embedding transformation
            features = self.mlp(image) + features

        if self.training:
            img_emb= features
            features_in = self.linear1(features)
            rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
            rand_list_2 = torch.rand(features.size(0), features.size(1)).to(features.device)
            mask1 =(rand_list_1 >= 0.2).unsqueeze(-1)
            mask2 = (rand_list_2 >= 0.2).unsqueeze(-1)

            feature_1 = features_in.masked_fill(mask1 == 0,-10000)
            features_k_softmax1= nn.Softmax(dim=1)(feature_1-torch.max(feature_1,dim=1)[0].unsqueeze(1))
            attn1 = features_k_softmax1.masked_fill(mask1 == 0,0)
            feature_img1 = torch.sum(attn1 * img_emb,dim=1)

            feature_2 = features_in.masked_fill(mask2 == 0,-10000)
            features_k_softmax2= nn.Softmax(dim=1)(feature_2-torch.max(feature_2,dim=1)[0].unsqueeze(1))
            attn2 = features_k_softmax2.masked_fill(mask2 == 0,0)
            feature_img2 = torch.sum(attn2 * img_emb,dim=1)

            feature_img = torch.cat((feature_img1.unsqueeze(1),feature_img2.unsqueeze(1)),dim=1).reshape(-1,img_emb.size(-1))#2b，d

        else:
            img_emb= features
            features_in = self.linear1(features)

            attn = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1))
            feature_img = torch.sum(attn * img_emb,dim=1)

        if not self.no_imgnorm:
            feature_img = l2norm(feature_img, dim=-1)

        return feature_img

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        features = self.image_encoder(base_features)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('/root/.cache/torch/transformers/bert-base-uncased/')
        self.linear = nn.Linear(768, embed_size)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D

        cap_emb = self.linear(bert_emb)

        cap_emb = self.dropout(cap_emb)

        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        cap_emb = cap_emb[:, :int(lengths.max()), :] 
        features_in = self.linear1(cap_emb)
        features_in = features_in.masked_fill(mask == 0,-10000)
        features_k_softmax = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1))
        attn = features_k_softmax.masked_fill(mask == 0,0)
        feature_cap = torch.sum(attn * cap_emb,dim=1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)

        return feature_cap

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
