import torch
from torch import nn
from torch.nn import functional

import torchvision
from torchvision import models

import numpy as np
import logging

EPSILON = 1e-12


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = functional.interpolate(attentions, size=(H, W), mode='bilinear')

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = functional.normalize(feature_matrix, dim=-1)
        return feature_matrix


"""Create BasicConv2d Layer and also perform a batch normalization operation."""

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return functional.relu(x, inplace=True)


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, model='inception', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.model = model

        # Network Initialization
        if 'densenet121' in model:
            pretrain = models.densenet121(
                #weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
                weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)

            modules = list(list(pretrain.children())[0])[:-2]
            self.features = nn.Sequential(*modules)
            self.num_features = 512
        elif 'inception' in model:
            # @@ TODO - use `weights=` instead of deprecated `pretrained=`
            self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
            self.num_features = 768
        elif 'vgg' in model:
            # @@ TODO - use `weights=` instead of deprecated `pretrained=`
            pretrain = models.vgg16(pretrained=pretrained)
            modules = list(pretrain.children())[:-2]
            self.features = nn.Sequential(*modules)
            self.num_features = 512
        elif 'resnet34' in model or 'resnet50' in model:
            if 'resnet34' in model:
                pretrain = models.resnet34(
                    weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            elif 'resnet50' in model:
                pretrain = models.resnet50(
                    weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

            modules = list(pretrain.children())[:-2]  # delete the last fc layer.
            self.features = nn.Sequential(*modules)
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported model: %s' % model)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(model, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        # if self.model != 'inception_mixed_7c':
        if self.model != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = functional.normalize(attention_weights, p=1, dim=0)
                # Randomly picked out two.
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            # In the testing case, it will avrage all the attention(combine) into single attention map
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: ✨ All params loaded' % type(self).__name__)
        else:
            msg = '⚠️ Some params were not loaded'
            logging.info(f'%s: {msg}:' % type(self).__name__)

            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('  %s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

            if strict: raise ValueError(msg)

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)
