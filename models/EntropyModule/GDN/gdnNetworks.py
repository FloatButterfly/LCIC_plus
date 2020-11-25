import torch
import torch.nn as nn
import torch.nn.functional as F
from .gdn import GDN2d
from .SingleConv2d import SingleConv2d


class NetModule(nn.Module):
    """ Base module.
    """
    def __init__(self, in_channels, latent_channels, out_channels):
        super(NetModule, self).__init__()
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)


class Encoder_GDN(NetModule):
    """Encoder with GDN activation.
    TODO: this should implemented with cross correlation.
    """
    def __init__(self, in_channels, latent_channels, out_channels):  # 1,128,128
        super(Encoder_GDN, self).__init__(in_channels, latent_channels, out_channels)

        self._model = nn.Sequential(
            SingleConv2d(self._nic, self._nlc, 5, 2, upsample=False, use_bias=True),  # 1,128,5,2
            GDN2d(self._nlc, inverse=False),
            SingleConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SingleConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SingleConv2d(self._nlc, self._noc, 5, 2, upsample=False, use_bias=True)
        )

    def forward(self, x):
        return self._model(x)


class Decoder_GDN(NetModule):
    """Decoder with IGDN activation.
    """
    def __init__(self, in_channels, latent_channels, out_channels):
        super(Decoder_GDN, self).__init__(in_channels, latent_channels, out_channels)

        self._model = nn.Sequential(
            SingleConv2d(self._nic, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SingleConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SingleConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SingleConv2d(self._nlc, self._noc, 5, 2, upsample=True, use_bias=True)
        )

    def forward(self, x):
        return self._model(x)
