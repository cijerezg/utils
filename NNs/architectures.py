"""Different NN architectures."""

import torch.nn as nn
import torch
import pdb


class MLP(nn.Module):
    """
    Implement an MLP.

    Parameters
    ----------
    inp: integer
       Input dimension
    out: integer
       Output dimension
    layers: list
       Each element contains the width of each layer. Length is
       the number of layers.
    sigma: string
       "relu" or "leaky-relu"
    hidden: bool (default: False)
       If True, it returns the output of hidden layers

    Returns
    -------
    tensor: torch tensor
       The output of neural network and if hidden is set to True,
       then it also returns the output of hidden layers.
    """

    def __init__(self, inp, out, layers, sigma, out_sigma, hidden=False, bias=True):
        """Initialize for NN."""
        super(MLP, self).__init__()
        self.layers = layers
        self.fcin = nn.Linear(inp, layers[0], bias=True)
        self.fcs = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1], bias=bias) for i in range(len(layers) - 1)])
        self.fcout = nn.Linear(layers[-1], out, bias=True)
        if sigma == "relu":
            self.sigma = nn.ReLU()
        elif sigma == "lrelu":
            self.sigma = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.sigma = nn.Identity()
        if out_sigma == 'tanh':
            self.out_sigma = nn.Tanh()
        else:
            self.out_sigma = nn.Identity()
        self.hidden = hidden

    def forward(self, x):
        """Forward pass of neural network."""
        hidden = []
        x = self.sigma(self.fcin(x))
        if self.hidden is True:
            hidden.append(x.detach().clone())
        for i in range(len(self.layers)-1):
            x = self.sigma(self.fcs[i](x))
            if self.hidden is True:
                hidden.append(x.detach().clone())
        x = self.out_sigma(self.fcout(x))
        return x#, hidden


class CNN_1D(nn.Module):
    """
    Implement CNN.

    Parameters
    ----------
    inp: integer
       Input dimension
    out: integer
       Output dimension
    chan: integer
       Number of channels
    ll: list
       Each element contains the width of each layer. Length is
       the number of layers.
    sigma: string
       "relu" or "leaky-relu"
    hidden: bool (default: False)
       If True, it returns the output of hidden layers

    Returns
    -------
    tensor: torch tensor
       The output of neural network and if hidden is set to True,
       then it also returns the output of hidden layers.
    """
    
    def __init__(self, inp, out, chan, ll, sigma, hidden=False, bias=False):
        """Initialize NN."""
        super(CNN_1D, self).__init__()
        self.ll = ll
        self.fcin = nn.Conv1d(chan, ll[0], 3, padding=1)
        self.fcs = nn.ModuleList(
            [nn.Conv1d(ll[i], ll[i+1], 3, bias=bias, padding=1) for i in range(len(ll) - 1)])
        self.fcout = nn.Linear(ll[-1] * inp, out, bias=False)
        if sigma == "relu":
            self.sigma = nn.ReLU()
        elif sigma == "lrelu":
            self.sigma = nn.LeakyReLU()
        else:
            self.sigma = nn.Identity()
        self.hidden = hidden
    
    def forward(self, x):
        """Forward pass of CNN."""
        hidden = []
        x = x.unsqueeze(1)
        x = self.sigma(self.fcin(x))
        for i in range(len(self.ll)-1):
            x = self.sigma(self.fcs[i](x))
            if self.hidden is True:
                hidden.append(x.detach().clone())
        x = self.fcout(x.view(x.shape[0], -1))
        return x, hidden
