import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv2d(nn.Module):
    """A wrapper class for Conv2d and its upsampling operation.
    Kernel size and stride size are both 2-d and have the same value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample, use_bias):
        super(SingleConv2d, self).__init__()
        # Now only check these combinations of parameters.
        if (kernel_size, stride) not in [(9, 4), (5, 2), (3, 1)]:
            raise ValueError("This pair of parameters (kernel_size, stride) has not been checked!")

        # Save the input parameters.
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _get_pair(kernel_size)
        self.stride = _get_pair(stride)
        self.upsample = bool(upsample)
        self.use_bias = bool(use_bias)

        # Define the parameters.
        if self.upsample:
            self.weight = nn.Parameter(torch.Tensor(
                self.in_channels, self.out_channels, *self.kernel_size
            ))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, *self.kernel_size
            ))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight)
            if self.bias is not None:
                self.bias.zero_()

    # 编码是：补零(h,w)->(0,1)+卷积，解码对称的过程是：反卷积+裁剪
    # TODO: 这里没有补零 是在输入之前补零的么？
    def forward(self, inputs):
        if self.upsample:
            outputs = F.conv_transpose2d(inputs, self.weight, self.bias, self.stride, padding=0)
            outputs = outputs[
                      slice(None), slice(None),
                      self.kernel_size[0] // 2 : self.kernel_size[0] // 2 - (self.kernel_size[0] - self.stride[0]),
                      self.kernel_size[1] // 2 : self.kernel_size[1] // 2 - (self.kernel_size[1] - self.stride[1])
                      ]
        else:
            outputs = F.conv2d(inputs, self.weight, self.bias, self.stride, tuple(k // 2 for k in self.kernel_size))

        return outputs


def _get_pair(inputs):
    """
    Return a tuple of (int, int).
    """
    if isinstance(inputs, int):
        outputs = (inputs, inputs)
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) != 2:
            raise ValueError("Length of parameters should be TWO!")
        else:
            outputs = tuple(int(item) for item in inputs)
    else:
        raise TypeError("Not proper type!")

    return outputs
