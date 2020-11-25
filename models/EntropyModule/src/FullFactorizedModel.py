import math

import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from EntropyUtils import Quantizer, MIN_LIKELIHOOD
from math import lower_bound, signum

import pdb

class HyperpriorDensity(nn.Module):
    def __init__(self, n_channels, filters=(3, 3, 3), init_scale=10.):  # 网络：(1,3,3,3,1)
        super(HyperpriorDensity, self).__init__()
        self.n_channels = int(n_channels)
        self._ft = (1,) + tuple(int(nf) for nf in filters) + (1,)
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Define univariate density model
        for k in range(len(self.filters) + 1):
            # Weights
            H_init = np.log(np.expm1(1 / scale / filters[k + 1]))
            H_k = nn.Parameter(torch.ones((n_channels, filters[k + 1], filters[k])))  # apply softmax for non-negativity
            torch.nn.init.constant_(H_k, H_init)
            self.register_parameter('H_{}'.format(k), H_k)

            # Scale factors
            a_k = nn.Parameter(torch.zeros((n_channels, filters[k + 1], 1)))
            self.register_parameter('a_{}'.format(k), a_k)

            # Biases
            b_k = nn.Parameter(torch.zeros((n_channels, filters[k + 1], 1)))
            torch.nn.init.uniform_(b_k, -0.5, 0.5)
            self.register_parameter('b_{}'.format(k), b_k)

    def cdf_logits(self, x, update_parameters=True):
        """Evaluate logits of the cummulative densities.
        Args: 
            x: torch.Tensor, [C, 1, *]
        """
        logits = x
        for k in range(len(self.filters) + 1):
            H_k = getattr(self, 'H_{}'.format(str(k)))  # Weight
            a_k = getattr(self, 'a_{}'.format(str(k)))  # Scale
            b_k = getattr(self, 'b_{}'.format(str(k)))  # Bias

            if update_parameters is False:
                H_k, a_k, b_k = H_k.detach(), a_k.detach(), b_k.detach()
            logits = torch.bmm(F.softplus(H_k), logits)  # [C,filters[k+1],*]
            logits = logits + b_k
            logits = logits + torch.tanh(a_k) * torch.tanh(logits)

        return logits

    def forward(self, inputs, update_parameters):
        # Inputs shape => [Channels, 1, Values]
        return self.cdf_logits(inputs, update_parameters)


class FullFactorizedModel(nn.Module):
    def __init__(self, n_channels, filters, min_likelihood=MIN_LIKELIHOOD, sign_reversal=True):
        super(FullFactorizedModel, self).__init__()
        self.n_channels = int(n_channels)
        self.min_likelihood = float(min_likelihood)
        self._sign_reversal = sign_reversal

        self.cdf_logits = HyperpriorDensity(self.n_channels, filters)
        self.quantizer = Quantizer()

        # Define the "optimize_integer_offset".
        self.quantiles = nn.Parameter(torch.zeros(self.n_channels, 1, 1))
        self.register_buffer("target", torch.zeros(self.n_channels, 1, 1))

    def likelihood(self, x, shape, **kwargs):
        """
        Expected input: [C, 1, *]
        """
        latents = x

        # Evaluate densities.
        cdf_lower = self.cdf_logits(latents - 0.5, update_parameters=False)
        cdf_upper = self.cdf_logits(latents + 0.5, update_parameters=False)
        if self._sign_reversal:
            # Numerical stability using some sigmoid identities
            # to avoid subtraction of two numbers close to 1
            sign = signum(cdf_lower + cdf_upper).detach()
            likelihood = torch.abs(
                torch.sigmoid(-sign * cdf_upper) - torch.sigmoid(-sign * cdf_lower)  # sign*num 相当于取绝对值
            )
        else:
            likelihood = torch.sigmoid(cdf_upper) - torch.sigmoid(cdf_lower)
        
        if self.min_likelihood > 0:
            likelihood = lower_bound(likelihood, self.min_likelihood)
        
        # Reshape to [N, C, H, W]
        likelihood = torch.reshape(likelihood, shape)
        likelihood = torch.transpose(likelihood, 0, 1)
        return likelihood

    def forward(self, x):
        """
        Expected input: [N, C, H, W]
        """
        latents = x
        
        # Convert latents to [C, 1, *] format.
        latents = torch.transpose(latents, 0, 1)
        shape = latents.shape
        values = latents.contiguous().view(self.n_channels, 1, -1)

        # Add noise or quantize.
        values = self.quantizer(values, self.training, self.quantiles)

        likelihood = self.likelihood(values, shape)
        
        # Reshape to [N, C, H, W]
        values = values.view(*shape)
        values = torch.transpose(values, 0, 1)

        return values, likelihood

    def integer_offset(self):
        logits = self.cdf_logits(self.quantiles, update_parameters=True)
        loss = torch.sum(torch.abs(logits - self.target))

        return loss

    def visualize(self, index, minval=-10, maxval=10, interval=0.1):
        # Get the default dtype and device.
        var = next(self.parameters())
        dtype, device = var.dtype, var.device

        # Compute the density.
        x = torch.arange(minval, maxval, interval, dtype=dtype, device=device)
        x_ = torch.zeros(self.n_channels, 1, len(x), dtype=dtype, device=device, requires_grad=True)
        with torch.no_grad():
            x_[index, 0, :] = x

        w_ = torch.sigmoid(self._cdf(x_, update_parameters=True))
        w_.backward(torch.ones_like(w_))
        y = x_.grad[index, 0, :]

        # Convert the tensor to numpy array.
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        plt.figure()
        plt.plot(x, y, 'r-')
        plt.show()

        return x, y



if __name__ == "__main__":
    n_channels = 64
    filters = (3, 3, 3)
    fullFactorizedModel = FullFactorizedModel(n_channels, filters)
    pdb.set_trace()

    input = torch.randn((2, n_channels, 110, 110))
    likelihood1 = fullFactorizedModel(input)
    