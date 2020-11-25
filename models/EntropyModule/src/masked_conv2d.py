import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        # Check the input parameter
        if mask_type not in {'A', 'B'}:
            raise TypeError("Parameter 'mask_type' should be in \{'A', 'B'\}.")

        # Define the mask parameter
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kH, kW = self.mask.shape
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:, :] = 0

    def forward(self, inputs):
        # Mask the weight
        # print(self.weight.shape,self.mask.shape)
        with torch.no_grad():
            self.weight *= self.mask

        return super(MaskedConv2d, self).forward(inputs)
