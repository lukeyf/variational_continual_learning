import torch
import torch.nn as nn
import torch.nn.functional as F

class MFVI_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MFVI_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean and log variance parameters for weights and biases
        self.W_m = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_m = nn.Parameter(torch.Tensor(out_features))
        self.W_logv = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_logv = nn.Parameter(torch.Tensor(out_features))

        # Prior distributions initialized to None
        self.prior_W_m = None
        self.prior_b_m = None
        self.prior_W_logv = None
        self.prior_b_logv = None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mean parameters to a normal distribution and log variance parameters to a small value
        self.W_m.data.normal_(0, 0.1)
        self.b_m.data.normal_(0, 0.1)
        self.W_logv.data.fill_(-6.0)
        self.b_logv.data.fill_(-6.0)
        self.set_priors()

    def set_priors(self):
        # Set priors to the current parameters
        self.prior_W_m = self.W_m.detach().clone()
        self.prior_b_m = self.b_m.detach().clone()
        self.prior_W_logv = self.W_logv.detach().clone()
        self.prior_b_logv = self.b_logv.detach().clone()

    def forward(self, x, sample=True):
        # Calculating the standard deviations from log variances
        W_std = torch.exp(0.5 * self.W_logv)
        b_std = torch.exp(0.5 * self.b_logv)
        
        # Calculating the mean and variance of the activation
        act_mu = F.linear(x, self.W_m, self.b_m)
        act_var = 1e-16 + F.linear(x.pow(2), W_std.pow(2)) + b_std.pow(2)
        act_std = torch.sqrt(act_var)

        # Sampling from the distribution if training or requested
        if self.training or sample:
            eps = torch.randn_like(act_mu)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_divergence(self, device):
        self.update_prior_device(device)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculating the standard deviation for posterior and prior
        W_std_post = torch.exp(0.5 * self.W_logv)
        b_std_post = torch.exp(0.5 * self.b_logv)
        W_std_prior = torch.exp(0.5 * self.prior_W_logv)
        b_std_prior = torch.exp(0.5 * self.prior_b_logv)

        # KL divergence for weights
        kl_div_W = torch.log(W_std_prior / W_std_post) + \
                   ((W_std_post**2 + (self.W_m - self.prior_W_m)**2) / (2 * W_std_prior**2)) - 0.5
        kl_div_W = kl_div_W.sum()

        # KL divergence for biases
        kl_div_b = torch.log(b_std_prior / b_std_post) + \
                   ((b_std_post**2 + (self.b_m - self.prior_b_m)**2) / (2 * b_std_prior**2)) - 0.5
        kl_div_b = kl_div_b.sum()

        # Total KL divergence
        total_kl = kl_div_W + kl_div_b

        return total_kl / num_params

    def update_prior_device(self, device):
        # Ensure priors are on the correct device
        self.prior_W_m = self.prior_W_m.to(device)
        self.prior_b_m = self.prior_b_m.to(device)
        self.prior_W_logv = self.prior_W_logv.to(device)
        self.prior_b_logv = self.prior_b_logv.to(device)

class MFVI_NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_tasks=1, single_head=False):
        super(MFVI_NN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_tasks = num_tasks
        self.single_head = single_head
        self.layers = nn.ModuleList()
        self.task_specific_layers = nn.ModuleDict()  # Using ModuleDict for task-specific layers

        # Construct shared layers
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            self.layers.append(MFVI_Layer(sizes[i], sizes[i + 1]))

        # Construct task-specific output layers
        if single_head:
            self.task_specific_layers[str(0)] = MFVI_Layer(sizes[-1], output_size)
        else:
            for task_id in range(num_tasks):
                self.task_specific_layers[str(task_id)] = MFVI_Layer(sizes[-1], output_size)

    def forward(self, x, task_id=0, sample=True):
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample))
        if self.single_head:
            x = self.task_specific_layers["0"](x, sample=sample)
        else:
            x = self.task_specific_layers[str(task_id)](x, sample=sample)
        return x

    def kl_divergence(self):
        kl_div = 0
        # Accumulate KL divergence from shared and task-specific layers
        for layer in self.layers:
            kl_div += layer.kl_divergence(next(self.parameters()).device)
        for task_layer in self.task_specific_layers.values():
            kl_div += task_layer.kl_divergence(next(self.parameters()).device)
        return kl_div

    def update_priors(self):
        # Update priors in each layer
        for layer in self.layers + list(self.task_specific_layers.values()):
            layer.set_priors()
