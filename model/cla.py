import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from copy import deepcopy

class MFVI_NN(nn.Module):
    """Bayesian Neural Network with Mean Field Variational Inference"""
    def __init__(self, input_size, hidden_sizes, output_size, no_train_samples=10, no_pred_samples=100, prior_mean=0, prior_var=1):
        super(MFVI_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        # Define layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(VariationalLayer(sizes[i], sizes[i+1], prior_mean, prior_var))

    def forward(self, x, samples=1):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, samples))
        x = self.layers[-1](x, samples)
        return x

    def kl_divergence(self):
        # Calculate the KL divergence term
        kl = 0
        for layer in self.layers:
            weight_kl = 0.5 * torch.sum(
                layer.prior_logvar - layer.weight_logvar + (layer.weight_logvar.exp() + (layer.weight_mu - layer.prior_mean) ** 2) / layer.prior_var.exp() - 1)
            bias_kl = 0.5 * torch.sum(
                layer.prior_logvar[0,0] - layer.bias_logvar + (layer.bias_logvar.exp() + (layer.bias_mu - layer.prior_mean[0,0]) ** 2) / layer.prior_var.exp()[0,0] - 1)
            kl += weight_kl + bias_kl
        return kl