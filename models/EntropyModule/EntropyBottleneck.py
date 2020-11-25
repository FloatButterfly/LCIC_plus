import pdb

import torch
import torch.nn as nn
import numpy as np

from GDN.SignConv2d import SignConv2d
from src.EntropyUtils import GaussianModel, Quantizer
from srcFullFactorizedModel import FullFactorizedModel
from src.masked_conv2d import MaskedConv2d


class EdgeEntropyBottleneck(nn.Module):
    """Entropy bottleneck for edge imgae.
    """
    def __init__(self, n_channels):
        super(EdgeEntropyBottleneck, self).__init__()
        self.n_channels = int(n_channels)

        self.hyper_encoder = nn.Sequential(
            SignConv2d(self.n_channels, self.n_channels, 3, 1, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self.n_channels, self.n_channels, 5, 2, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self.n_channels, self.n_channels, 5, 2, upsample=False, use_bias=True)
        )

        self.hyper_decoder = nn.Sequential(
            SignConv2d(self.n_channels, self.n_channels, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self.n_channels, self.n_channels * 3 // 2, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self.n_channels * 3 // 2, self.n_channels * 2, 3, 1, upsample=True, use_bias=True)
        )

        self.context_model = MaskedConv2d('A', self.n_channels, self.n_channels * 2, 5, padding=2)

        self.hyper_parameter = nn.Sequential(
            nn.Conv2d(self.n_channels * 4, self.n_channels * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_channels * 10 // 3, self.n_channels * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_channels * 8 // 3, self.n_channels * 2, 1)
        )

        filters = (3, 3, 3)
        MIN_SCALE = 1e-1
        MIN_LIKELIHOOD = 1e-9
        
        self.quantizer = Quantizer()
        self.fullFactorizedModel = FullFactorizedModel(self.n_channels, filters, MIN_LIKELIHOOD, True)
        self.gaussianModel = GaussianModel(MIN_SCALE, MIN_LIKELIHOOD, True)

    def forward(self, y):
        z = self.hyper_encoder(y)                                       # 求先验
        z_hat, z_likelihoods = self.fullFactorizedModel(z)              # 先验概率估计 train=>z_tilde, test=>z_hat
        u = self.hyper_decoder(z_hat)                                   # 先验信息解码
        y_hat = self.quantizer(y, self.training)                        # 量化  train=>y_tilde, test=>y_hat
        v = self.context_model(y_hat)                                   # 上下文模型卷积
        # if u.shape != v.shape:
        #     print(z_hat.shape, u.shape, v.shape, y.shape)
        parameters = self.hyper_parameter(torch.cat((u, v), dim=1))     # 上下文与先验融合求概率分布参数
        mean, scale = parameters.split(self.n_channels, dim=1)          # 参数分成 方差，均值
        y_likelihoods = self.gaussianModel(y_hat, mean, scale)          # 高斯模型求概率

        len_bits = torch.sum(-torch.log2(z_likelihoods)) + torch.sum(-torch.log2(y_likelihoods))    # 求出总熵

        return y_hat, z_hat, len_bits

    @property
    def offset(self):
        return self.fullFactorizedModel.integer_offset()


class TextureEntropyBottleneck(nn.Module):
    """Entropy bottleneck for texture vector.
    """
    def __init__(self, n_channels):
        super(TextureEntropyBottleneck, self).__init__()
        self.n_channels = n_channels        

        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 4, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_channels // 4, n_channels // 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels // 32, kernel_size=1, stride=1)
        )

        self.hyper_decoder = nn.Sequential(
            nn.Conv2d(n_channels // 32, n_channels // 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels // 2, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_channels // 2, n_channels * 2, kernel_size=1, stride=1)
        )

        # self.context_model = MaskedConv2d('A', n_channels, n_channels * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(n_channels * 4, n_channels * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(n_channels * 10 // 3, n_channels * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(n_channels * 8 // 3, n_channels * 2, 1)
        # )
        filters = (3, 3, 3)
        MIN_SCALE = 1e-1
        MIN_LIKELIHOOD = 1e-9

        self.quantizer = Quantizer()
        self.fullFactorizedModel = FullFactorizedModel(n_channels // 32, filters=filters, min_likelihood=MIN_LIKELIHOOD, sign_reversal=True)
        self.gaussianModel = GaussianModel(MIN_SCALE, MIN_LIKELIHOOD, True)

    def forward(self, y, verbose=False):
        z = self.hyper_encoder(y)                                   # 求先验 [bs,16,19,1]
        z_hat, z_likelihoods = self.fullFactorizedModel(z)          # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)                               # 先验信息解码
        y_hat = self.quantizer(y, self.training)                    # 量化
        # v = self.context_model(y_hat)                             # 上下文模型卷积
        # if u.shape != v.shape:
        #     print(z_hat.shape, u.shape, v.shape, y.shape)
        # parameters = self.hyper_parameter(torch.cat((u, v), dim=1))  # 上下文与先验融合求概率分布参数
        
        mean, scale = u.split(self.n_channels, dim=1)               # 参数分成 方差，均值
        y_likelihoods = self.gaussianModel(y_hat, mean, scale)      # 高斯模型求概率
        
        len_bits = torch.sum(-torch.log2(z_likelihoods)) + torch.sum(-torch.log2(y_likelihoods))  # 求出总熵
        return y_hat, z_hat, len_bits

    @property
    def offset(self):
        return self.fullFactorizedModel.integer_offset()


if __name__ == "__main__":
    n_channels = 64
    textureEntropy = TextureEntropyBottleneck(n_channels)
    
    pdb.set_trace()
    input = torch.randn(3, n_channels, 110, 110)
    y_hat, z_hat, length = textureEntropy(input)

    print("finished")
