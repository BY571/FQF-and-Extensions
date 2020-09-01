import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from copy import copy




def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)

#class NoisyLinear(nn.Linear):
#    # Noisy Linear Layer for independent Gaussian Noise
#    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
#        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
#        # make the sigmas trainable:
#        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
#        # not trainable tensor for the nn.Module
#        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
#        # extra parameter for the bias and register buffer for the bias parameter
#        if bias: 
#            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
#            self.register_buffer("epsilon_bias", torch.zeros(out_features))
#    
#        # reset parameter as initialization of the layer
#        self.reset_parameter()
#    
#    def reset_parameter(self):
#        """
#        initialize the parameter of the layer and bias
#        """
#        std = math.sqrt(3/self.in_features)
#        self.weight.data.uniform_(-std, std)
#        self.bias.data.uniform_(-std, std)
#
#    
#    def forward(self, input):
#        # sample random noise in sigma weight buffer and bias buffer
#        self.epsilon_weight = torch.normal(self.epsilon_weight)
#        bias = self.bias
#        if bias is not None:
#            self.epsilon_bias = torch.normal(self.epsilon_bias)
#            bias = bias + self.sigma_bias * self.epsilon_bias
#        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)

class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for factorised Gaussian noise 
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # extra parameter for the bias 
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features,))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        # not trainable tensor for the nn.Module
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):  
        """
        initialize the parameter of the layer and bias
        """
        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def f(self, x):
        """
        calc  output noise for weights and biases

        bias could also be just x
        """
        return x.normal_().sign().mul(x.abs().sqrt())

    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

        weight = self.mu_weight + self.sigma_weight * self.eps_q.ger(self.eps_p)
        bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()

        return F.linear(input, weight, bias) 


class QVN(nn.Module):
    """Quantile Value Network"""
    def __init__(self, state_size, action_size,layer_size, n_step, device, seed, dueling=False, noisy=False, N=32):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(self.input_shape)
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.dueling = dueling
        self.device = device
        if noisy:
            layer = NoisyLinear
        else:
            layer = nn.Linear


        # Network Architecture
        if self.state_dim == 3:
            self.head = nn.Sequential(
                nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            )#.apply() #weight init
            self.cos_embedding = layer(self.n_cos, self.calc_input_layer())
            self.ff_1 = layer(self.calc_input_layer(), layer_size)
            self.cos_layer_out = self.calc_input_layer()

        else:   
            self.head = nn.Linear(self.input_shape[0], layer_size) 
            self.cos_embedding = nn.Linear(self.n_cos, layer_size)
            self.ff_1 = layer(layer_size, layer_size)
            self.cos_layer_out = layer_size
            if not noisy: weight_init([self.head, self.ff_1])
        if dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
            if not noisy: weight_init([self.ff_1])
        else:
            self.ff_2 = layer(layer_size, action_size)    
            if not noisy: weight_init([self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self,taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos
    
    def forward(self, input):
        """Calculate the state embeddings"""
        if self.state_dim == 3:
            x =  torch.relu(self.head(input))
            return x.view(input.size(0), -1)
        else:
            return torch.relu(self.head(input))
        
    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        
         """
        if embedding==None:
            x = self.forward(input)
            if self.state_dim == 3: x = x.view(input.size(0), -1)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)   
        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size)
    
    

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):
        """
        Calculates tau, tau_ and the entropy
        
        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """

        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy