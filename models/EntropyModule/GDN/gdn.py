import torch
import torch.nn as nn
import torch.nn.functional as F

from ..src.math import lower_bound


class NonnegativeParameterizer(nn.Module):
    def __init__(self, shape, initializer, minimum=0, reparam_offset=2**-18):
        super(NonnegativeParameterizer, self).__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        # Define some temporal variables
        self._pedestal = self.reparam_offset ** 2
        self._bound = (self.minimum + self.reparam_offset ** 2) ** .5

        # Define the real parameters
        self._var = nn.Parameter(torch.sqrt(initializer(shape) + self._pedestal))

    def forward(self):
        var = lower_bound(self._var, self._bound)
        var = torch.pow(var, 2) - self._pedestal

        return var


class GDN2d(nn.Module):
    """ Generalized diverse normalization (GDN) layer.
    Implements an activation function that is essentially a multivariate
    generalization of a particular sigmoid-type function:

        y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    
    where `i` and `j` run over channels. This implementation never sums across
    spatial dimensions. It is similar to local response normalization, but much
    more flexible, as `beta` and `gamma` are trainable parameters.
    """
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=.1):
        super(GDN2d, self).__init__()
        self.channels = int(channels)  # 128
        self.inverse = bool(inverse)

        self._beta = NonnegativeParameterizer(
            shape=(self.channels,),
            initializer=lambda shape: torch.ones(*shape),
            minimum=beta_min
        )
        self._gamma = NonnegativeParameterizer(
            shape=(self.channels, self.channels),
            initializer=lambda shape: torch.eye(*shape) * gamma_init
        )

    @property
    def beta(self):
        return self._beta()

    @property
    def gamma(self):
        return self._gamma().view(self.channels, self.channels, 1, 1)

    def forward(self, inputs):
        norm_pool = F.conv2d(inputs ** 2, self.gamma, self.beta)

        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)

        outputs = inputs * norm_pool

        return outputs


if __name__ == "__main__":
    n_batch = 3
    n_channel = 64
    gdn = GDN2d(n_channel, False)

    input = torch.randn(n_batch, n_channel, 128, 128)
    output = gdn(input)
    print(output.shape)
