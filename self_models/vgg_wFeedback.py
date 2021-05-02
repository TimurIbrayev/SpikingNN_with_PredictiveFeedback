#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:50:08 2021

@source: Nitin Rathi

@author: tibrayev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math


cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'VGG5_wFeedback_narrow' : [[32, 'D', 64, 64, 'D'], ['U', 64, 'drop', 32, 'U', 'in']],
    'VGG5_wFeedback' : [[64, 'D', 128, 128, 'D'], ['U', 128, 'drop', 64, 'U', 'in']],
    'VGG9_wFeedback' : [[64, 'D', 128, 256, 'D', 256, 512, 'D', 512, 'D', 512], 
                        [512, 'U', 512, 'U', 256, 'drop', 256, 'U', 128, 'drop', 64, 'U', 'in']],
}


class VGG_wFeedback(nn.Module):
    def __init__(self, vgg_name='VGG16', timesteps=1, labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2):
        super(VGG_wFeedback, self).__init__()
        
        self.vgg_name       = vgg_name
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.is_feedback    = True if 'wFeedback' in vgg_name else False
        if self.is_feedback:
            self.features, feature_dims   = self._make_layers(cfg[vgg_name][0])
            self.generate   = self._make_layers_feedback(cfg[vgg_name][1], feature_dims)
            self.timesteps  = timesteps
        else:
            self.features   = self._make_layers(cfg[vgg_name])
        
        if 'VGG5' in vgg_name and dataset!= 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(feature_dims*4*4, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif not 'VGG5' in vgg_name and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(feature_dims*2*2, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        if 'VGG5' in vgg_name and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(feature_dims*7*7, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif not 'VGG5' in vgg_name and dataset =='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(feature_dims*1*1, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )        
        self._initialize_weights2()
        

    def forward(self, x):
        if self.is_feedback:
            features = []
            predicts = []
            errors   = []
            for t in range(self.timesteps):
                if t == 0:
                    # forward t=0
                    # standard feedforward processing
                    forward_features    = self.features(x)
                    features.append(forward_features)
                else:
                    # forward t>0
                    # predictive coding processing (when the input signal is error signals)
                    forward_features    = self.features(error)                    
                    cumulative_features = features[-1] + forward_features
                    features.append(cumulative_features)
                    
                # backward
                # reconstructive processing, when the feedback tries to
                # reconstruct/predict the input based on cumulative features
                reconstructions     = self.generate(features[-1])
                predicts.append(reconstructions)
                
                error               = x - predicts[-1]
                errors.append(error**2)           

            # classifier processing
            # performed based on the last cumulative features
            out                 = features[-1].flatten(1)
            out                 = self.classifier(out)
            return out, predicts, errors
        
        else:
            # feedforward only processing (artifact of feedforward only ann model)
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    
    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers.pop()
                layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
                           nn.ReLU(inplace=True)
                           ]
                layers += [nn.Dropout(self.dropout)]           
                in_channels = x
        
        return nn.Sequential(*layers), in_channels


    def _make_layers_feedback(self, cfg, in_channels):
        layers = []
        
        for x in cfg:
            stride = 1

            if x == 'U':
                layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)]
            elif x == 'drop':
                layers += [nn.Dropout(self.dropout)]
            elif x == 'in':
                if self.dataset == 'MNIST':
                    x = 1
                    layers += [nn.ConvTranspose2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False)]
                    layers += [nn.Sigmoid()]
                else:
                    x = 3
                    layers += [nn.ConvTranspose2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False)]
                    layers += [nn.Tanh()]
            else:
                layers += [nn.ConvTranspose2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
                           nn.LeakyReLU()
                           ]       
                in_channels = x
        
        return nn.Sequential(*layers)



def test():
    for a in cfg.keys():
        if not 'wFeedback' in a:
            net = VGG_wFeedback(a)
            x = torch.randn(2,3,32,32)
            y = net(x)
            print(y.size())
        else:
            net     = VGG_wFeedback(a)
            x       = torch.randn(2,3,32,32)
            y, p    = net(x)
            print(y.size())
            print(p.size())
    # For VGG5 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG5')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
if __name__ == '__main__':
    test()
