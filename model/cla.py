import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from copy import deepcopy

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class VariationalLayer(nn.Module):
    """A simple variational layer with mean and log variance parameters for both the weights and the priors."""
    def __init__(self, input_features, output_features, prior_mean=0.0, prior_var=1.0):
        super(VariationalLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # Parameters for variational distribution
        self.weight_mu = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_mu = nn.Parameter(torch.Tensor(output_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(output_features))
        # Prior distribution parameters (not learnable)
        self.register_buffer('prior_mean', torch.full((output_features, input_features), prior_mean))
        self.register_buffer('prior_var', torch.full((output_features, input_features), prior_var))
        self.register_buffer('prior_logvar', torch.log(torch.full((output_features, input_features), prior_var)))
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.weight_logvar.data.fill_(-6)  # Start with a large variance
        self.bias_logvar.data.fill_(-6)

    def forward(self, x, samples=1):
        # Sample weights and biases using reparameterization
        weight = self.reparameterize(self.weight_mu, self.weight_logvar, samples)
        bias = self.reparameterize(self.bias_mu, self.bias_logvar, samples)
        return F.linear(x, weight, bias)

    def reparameterize(self, mu, logvar, samples):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size(), samples)
        return eps.mul(std).add_(mu)

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