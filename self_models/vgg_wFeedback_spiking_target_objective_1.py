#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:48:11 2021

@source: Nitin Rathi

@author: tibrayev
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 256, 'A', 512, 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'VGG5_wFeedback_narrow' : [[32, 'D', 64, 64, 'D'], ['U', 64, 'drop', 32, 'U', 'in']],
    'VGG5_wFeedback' : [[64, 'D', 128, 128, 'D'], ['U', 128, 'drop', 64, 'U', 'in']],
    'VGG9_wFeedback' : [[64, 'D', 128, 256, 'D', 256, 512, 'D', 512, 'D', 512], 
                        [512, 'U', 512, 'U', 256, 'drop', 256, 'U', 128, 'drop', 64, 'U', 'in']],    
}

class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*1.0).float(),torch.sign(input))
		return out

class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike):
        
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(input).cuda()
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad*grad_input, None

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class VGG_SNN_STDB_wFeedback(nn.Module):
    def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, alpha=0.3, beta=0.01, dropout=0.2, kernel_size=3, dataset='CIFAR10'):
        super().__init__()
        
        self.vgg_name           = vgg_name
        if activation == 'Linear':
            self.act_func       = LinearSpike.apply
        elif activation == 'STDB':
            self.act_func       = STDB.apply
        self.labels             = labels
        self.timesteps          = timesteps
        self.leak               = torch.tensor(leak)
        STDB.alpha              = alpha
        STDB.beta               = beta 
        self.dropout            = dropout
        self.kernel_size        = kernel_size
        self.dataset            = dataset
        self.input_layer        = PoissonGenerator()
        self.threshold          = {}
        self.mem                = {}
        self.mask               = {}
        self.spike              = {}
        self.epsilon            = None
        
        # Enabling feedback module
        self.is_feedback    = True if 'wFeedback' in vgg_name else False
        if self.is_feedback:
            self.features, feature_dims = self._make_layers(cfg[self.vgg_name][0])
            self.generate               = self._make_layers_feedback(cfg[self.vgg_name][1], feature_dims)
            self.classifier             = self._make_layers_classifier(feature_dims) 
        else:
            self.features, feature_dims = self._make_layers(cfg[self.vgg_name])
            self.classifier             = self._make_layers_classifier(feature_dims)


        self._initialize_weights2()
        
        
        # NOTE: with feedback path, there are two types of conv2d layers: 
        # one used with non-linearity and the other used instead of pooling without non-linearity                
        # in case of conv for downsampling (pooling), we need to reduce feature map size
        
        # Forward module
        prev = 0
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d) and (not self.features[l].kernel_size == (2,2)):
                self.threshold[prev+l]  = torch.tensor(default_threshold)
        
        # Backward module
        prev = len(self.features)
        for l in range(len(self.generate)):
            if isinstance(self.generate[l], nn.ConvTranspose2d) and (not self.generate[l].kernel_size == (2,2)):
                self.threshold[prev+l]  = torch.tensor(default_threshold)
				
        # Classifier module
        prev = len(self.features) + len(self.generate)
        for l in range(len(self.classifier)-1):
            if isinstance(self.classifier[l], nn.Linear):
                self.threshold[prev+l] 	= torch.tensor(default_threshold)

    def _initialize_weights2(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def threshold_update(self, scaling_factor=1.0, thresholds=[]):

		# Initialize thresholds
        self.scaling_factor = scaling_factor
		
        # Forward module
        prev = 0
        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d) and (not self.features[pos].kernel_size == (2,2)):
                if thresholds:
                    self.threshold[prev+pos]        = torch.tensor(thresholds.pop(0)*self.scaling_factor)
                #print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))
        
        # Backward module
        prev = len(self.features)
        for pos in range(len(self.generate)):
            if isinstance(self.generate[pos], nn.ConvTranspose2d) and (not self.generate[pos].kernel_size == (2,2)):
                if thresholds:
                    self.threshold[prev+pos]        = torch.tensor(thresholds.pop(0)*self.scaling_factor)

        # Classifier module
        prev = len(self.features) + len(self.generate)
        for pos in range(len(self.classifier)-1):
            if isinstance(self.classifier[pos], nn.Linear):
                if thresholds:
                    self.threshold[prev+pos]    = torch.tensor(thresholds.pop(0)*self.scaling_factor)
                #print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))


    def _make_layers(self, cfg):
        layers 		= []
        
        if self.dataset =='MNIST':
            in_channels = 1
        else:
            in_channels = 3

        for x in (cfg):
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
		
        features = nn.Sequential(*layers)
        return features, in_channels
    
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
    
    def _make_layers_classifier(self, in_channels):
        layers = []
        if 'VGG11' in self.vgg_name and self.dataset == 'CIFAR100':
            layers += [nn.Linear(in_channels*4*4, 1024, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(1024, 1024, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(1024, self.labels, bias=False)]
        elif 'VGG5' in self.vgg_name and self.dataset != 'MNIST':
            layers += [nn.Linear(in_channels*4*4, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, self.labels, bias=False)]
		
        elif not 'VGG5' in self.vgg_name and self.dataset != 'MNIST':
            layers += [nn.Linear(in_channels*2*2, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, self.labels, bias=False)]
		
        elif 'VGG5' in self.vgg_name and self.dataset == 'MNIST':
            layers += [nn.Linear(in_channels*7*7, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, self.labels, bias=False)]

        elif not 'VGG5' in self.vgg_name and self.dataset == 'MNIST':
            layers += [nn.Linear(in_channels*1*1, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, 4096, bias=False)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.5)]
            layers += [nn.Linear(4096, self.labels, bias=False)]
	

        classifer = nn.Sequential(*layers)
        return classifer
    

    def network_update(self, timesteps, leak):
        self.timesteps 	= timesteps
        self.leak 	 	= torch.tensor(leak)

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width 		= x.size(2)
        self.height 	= x.size(3)

        self.mem 		= {}
        self.mask 		= {}
        self.spike 		= {}			

        # Forward module
        prev = 0
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                if self.features[l].kernel_size == (2,2):
                    self.width              = self.width//self.features[l].kernel_size[0]
                    self.height             = self.height//self.features[l].kernel_size[1]
                else:
                    self.mem[prev+l]        = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)

            elif isinstance(self.features[l], nn.Dropout):
                self.mask[prev+l]           = self.features[l](torch.ones(self.mem[l-2].shape))
                        
            # with feedback path, we are no longer relying on AvgPool
            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width                  = self.width//self.features[l].kernel_size
                self.height                 = self.height//self.features[l].kernel_size


        # Backward module
        prev = len(self.features)
        for l in range(len(self.generate)):
            if isinstance(self.generate[l], nn.ConvTranspose2d):
                if self.generate[l].kernel_size == (2,2):
                    self.width              = self.width*self.generate[l].kernel_size[0]
                    self.height             = self.height*self.generate[l].kernel_size[1]
                else:
                    self.mem[prev+l]        = torch.zeros(self.batch_size, self.generate[l].out_channels, self.width, self.height)
            
            elif isinstance(self.generate[l], nn.Dropout):
                self.mask[prev+l]           = self.generate[l](torch.ones(self.mem[prev+l-2].shape))
                

        # Classifier module
        prev = len(self.features) + len(self.generate)
        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev+l]            = torch.zeros(self.batch_size, self.classifier[l].out_features)
			
            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev+l]           = self.classifier[l](torch.ones(self.mem[prev+l-2].shape))


        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)

    def zero_mean_unit_var_and_shiftminmax(self, input):
        # WARNING: Not backpropagatable!
        self.min                = 0.0
        self.max                = 1.0
        imgs                    = input.clone().detach()
        imgs_flat               = imgs.flatten(1)
        # zero mean unit variance
        per_img_mean            = imgs_flat.mean(dim=1, keepdim=True)
        per_img_std             = imgs_flat.std(dim=1, keepdim=True)
        imgs_flat_norm          = imgs_flat.sub_(per_img_mean).div_(per_img_std)
        # shift to range [min,max]
        per_img_min             = imgs_flat_norm.min(dim=1, keepdim=True)[0]
        per_img_max             = imgs_flat_norm.max(dim=1, keepdim=True)[0]
        imgs_flat_0to1          = ((imgs_flat_norm - per_img_min)/(per_img_max - per_img_min + 1e-5))*(self.max-self.min) - self.min
        return imgs_flat_0to1.view_as(input)

        
        
    def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
        self.neuron_init(x)
        max_mem=0.0
        
        recon_cumulative    = None
        predicts            = []
        errors              = []
        input_x             = x.clone().detach()
        count_spikes        = 0.0
		
        for t in range(self.timesteps):
            out_prev = self.input_layer(input_x)
			
            # Forward module
            for l in range(len(self.features)):
                if isinstance(self.features[l], (nn.Conv2d)) and (not self.features[l].kernel_size == (2,2)):
					
                    if find_max_mem and l==max_mem_layer:
                        print("looking for max mem on layer {} in features".format(l))
                        if (self.features[l](out_prev)).max()>max_mem:
                            max_mem = (self.features[l](out_prev)).max()
                        break

                    mem_thr             = (self.mem[l]/self.threshold[l]) - 1.0
                    out                 = self.act_func(mem_thr, (t-1-self.spike[l]))
                    rst                 = self.threshold[l]* (mem_thr>0).float()
                    self.spike[l]       = self.spike[l].masked_fill(out.bool(),t-1)
                    self.mem[l]         = self.leak*self.mem[l] + self.features[l](out_prev) - rst
                    out_prev            = out.clone()
                    count_spikes       += out.detach().clone().sum()

                elif isinstance(self.features[l], (nn.Conv2d)) and (self.features[l].kernel_size == (2,2)):
                    out_prev            = self.features[l](out_prev)

                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev            = self.features[l](out_prev)
				
                elif isinstance(self.features[l], nn.Dropout):
                    out_prev            = out_prev * self.mask[l]
                
            if find_max_mem and max_mem_layer<len(self.features):
                continue
            
            # Backward module
            out_prev_backward           = out_prev.clone() # not to alter the signal passed to classifier module
            #print("!:", out_prev_backward.requires_grad)
            prev = len(self.features)
            for l in range(len(self.generate)):
                if isinstance(self.generate[l], (nn.ConvTranspose2d)) and (not self.generate[l].kernel_size == (2,2)):
                    
                    if find_max_mem and (prev+l)==max_mem_layer:
                        print("looking for max mem on layer {} in generate".format(l))
                        if (self.generate[l](out_prev_backward)).max()>max_mem:
                            max_mem = (self.generate[l](out_prev_backward)).max()
                        break
                    
                    mem_thr             = (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
                    out                 = self.act_func(mem_thr, (t-1-self.spike[prev+l]))
                    #print("Out:", out.requires_grad)
                    rst                 = self.threshold[prev+l] * (mem_thr>0).float()
                    self.spike[prev+l]  = self.spike[prev+l].masked_fill(out.bool(),t-1)
                    self.mem[prev+l]    = self.leak*self.mem[prev+l] + self.generate[l](out_prev_backward) - rst
                    out_prev_backward   = out.clone()
                    count_spikes       += out.detach().clone().sum()
                
                elif isinstance(self.generate[l], (nn.ConvTranspose2d)) and (self.generate[l].kernel_size == (2,2)):
                    out_prev_backward   = self.generate[l](out_prev_backward)
                
                elif isinstance(self.generate[l], nn.Dropout):
                    out_prev_backward   = out_prev_backward * self.mask[prev+l]

            if find_max_mem and max_mem_layer<(len(self.features)+len(self.generate)):
                continue
            
            # Reconstruction part [what is the most optimal strategy???]
            #print("@:", out_prev_backward.requires_grad)
            if t == 0:
                recon_cumulative    = out_prev_backward.clone()
            else:
                recon_cumulative    = recon_cumulative + out_prev_backward
            #print(recon_cumulative.requires_grad)
            reconstructions     = self.generate[-1](recon_cumulative)
            #print(reconstructions.requires_grad)
            predicts.append(reconstructions)
            error               = x - predicts[-1]
            errors.append(error**2)
            
            # Activation Sparsity (target objective 1)
            # first, get which neurons are predicted within epsilon bound
            prediction_t        = ((predicts[-1] - 0.5)/(1.0-0.5))*(1.0-0.0) - 0.0 # FIXME: to compensate for offset (technical train-time error)
            difference_t        = (np.logical_and((prediction_t < x+self.epsilon).cpu(),(prediction_t > x-self.epsilon).cpu())).bool()
            print((difference_t*1.0).sum())
            # then, block these predicted neurons by masking their inputs to 0.0
            input_x[difference_t] = 0.0
            
            
            # Classifier module
            out_prev       	= out_prev.reshape(self.batch_size, -1)
            prev = len(self.features) + len(self.generate)
			
            for l in range(len(self.classifier)-1):													
                if isinstance(self.classifier[l], (nn.Linear)):
                    
                    if find_max_mem and (prev+l)==max_mem_layer:
                        print("looking for max mem on layer {} in classifier".format(l))
                        if (self.classifier[l](out_prev)).max()>max_mem:
                            max_mem = (self.classifier[l](out_prev)).max()
                        break

                    mem_thr             = (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
                    out                 = self.act_func(mem_thr, (t-1-self.spike[prev+l]))
                    rst                 = self.threshold[prev+l] * (mem_thr>0).float()
                    self.spike[prev+l]  = self.spike[prev+l].masked_fill(out.bool(),t-1)
                    self.mem[prev+l]    = self.leak*self.mem[prev+l] + self.classifier[l](out_prev) - rst
                    out_prev            = out.clone()
                    count_spikes       += out.detach().clone().sum()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev            = out_prev * self.mask[prev+l]
			
			# Compute the classification layer outputs
            if not find_max_mem:
                self.mem[prev+l+1]      = self.mem[prev+l+1] + self.classifier[l+1](out_prev)
        
        if find_max_mem:
            return max_mem
        else:
            return self.mem[prev+l+1], predicts, errors, count_spikes



