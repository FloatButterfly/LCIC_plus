import math
import torch
import torch.nn as nn
from math import lower_bound, signum
import maths
import pdb


MIN_SCALE = 0.11
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e4
TAIL_MASS = 2**(-8)

PRECISION_P = 16  # Precision of rANS coder

lower_bound_toward = maths.LowerBoundToward.apply


class Quantizer(nn.Module):
    def forward(self, x, is_training, offset=0):
        if is_training:
            y = x + torch.empty_like(x).uniform_(-0.5, 0.5)
        else:
            y = torch.round(x - offset) + offset

        return y


class GaussianModel(nn.Module):
    """Conditional Gaussian entropy model.

    Probability model for latents y. Based on Sec. 3. of [1].
    Returns convolution of Gaussian / logistic latent density with parameterized 
    mean and variance with 'boxcar' uniform distribution U(-1/2, 1/2).

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
    arXiv:1802.01436 (2018).
    """
    def __init__(self, scale_lower_bound=MIN_SCALE, min_likelihood=MIN_LIKELIHOOD, sign_reversal=True):
        super(GaussianModel, self).__init__()
        # Save the input parameters
        self.scale_lower_bound = float(scale_lower_bound)
        self.min_likelihood = float(min_likelihood)
        self.sign_reverse = bool(sign_reversal)

    def _likelihood(self, values, scale):
        denominator = lower_bound(scale, self.scale_lower_bound) * math.sqrt(2.0)

        # Evaluate the standardized cumulative density, assume 1 - c(x) = c(-x)
        # upper = efrc(-.5*(.5-values)/(scale*2**.5))
        # lower = efrc(-.5*(-.5-values)/(scale*2**.5))
        lower = values - 0.5
        upper = values + 0.5
        if not self.sign_reverse:
            likelihood = 0.5 * (torch.erf(upper / denominator) - torch.erf(lower / denominator))
        else:
            sign = signum(values).detach()
            likelihood = 0.5 * torch.abs(
                torch.erf(-sign * upper / denominator) - torch.erf(-sign * lower / denominator)
            )
        if self.min_likelihood > 0:
            likelihood = lower_bound(likelihood, self.min_likelihood)
        
        return likelihood

    def forward(self, inputs, mean, scale):
        values = inputs - mean

        return self._likelihood(values, scale)


if __name__ == "__main__":
    in_channels = 64
    min_likelihood = 1e-9
    min_scale = 1e-1
    gaussianModel = GaussianModel(scale_lower_bound=min_scale, min_likelihood=min_likelihood, sign_reversal=True)
    
    x = torch.randn(in_channels, 128)
    mean = torch.tensor(0.5)
    scale = torch.tensor(2)

    likelihood = gaussianModel(x, mean, scale)
    print(likelihood.shape)
