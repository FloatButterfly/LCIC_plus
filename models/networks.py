import functools
import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
from torchvision import models


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_G(opt, gpu_ids=[]):

    netG = opt.netG
    init_type = opt.init_type
    where_add = opt.where_add
    net = None

    if netG == "progressive_256" and where_add == 'AdaIN':
        net = Progressive_generator(opt)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu',
             use_sigmoid=False, init_type='xavier', num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                             use_sigmoid=use_sigmoid, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_E(input_nc, ndf, netE, output_nc=1, norm='batch', label_nc=0,
             init_type='xavier', gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)

    if netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)
    
    return init_net(net, init_type, gpu_ids)


def define_Feature_Net(requires_grad=False, net_type='vgg16', gpu_ids=[]):
    netFeature = None

    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())

    if net_type == 'vgg16':
        netFeature = Vgg16(requires_grad=requires_grad)
    else:
        raise NotImplementedError('Feature net name [%s] is not recognized' % net_type)

    if len(gpu_ids) > 0:
        netFeature.cuda(gpu_ids[0])

    return netFeature


# feature loss network
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        os.environ['TORCH_HOME'] = '../../../models/'
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf_i = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf_i, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                            stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                                             kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    # model[0] output 30*30*1 model[1] output 6*6*1
    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        mean = 0.0
        for i in range(self.num_D):
            output = self.model[i](down)
            mean += output.mean()
            result.append(output)
            if i != self.num_D - 1:
                down = self.down(down)  # 256*256*3 -> 128*128*3
        mean = mean / self.num_D * 1.0
        return result, mean


class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_loss, self).forward(img1, img2))


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
        self.gan_mode = gan_mode
        # self.register_buffer('real_label', torch.tensor(target_real_label))
        # self.register_buffer('fake_label', torch.tensor(target_fake_label))
        # self.loss = nn.MSELoss() if mse_loss else nn.BCELoss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).cuda()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real).cuda()
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)



def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'add_output_padding':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1, output_padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv



def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                                         stride=1, padding=padw, bias=True))]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [spectral_norm(nn.Conv2d(inplanes, outplanes,
                                         kernel_size=1, stride=1, padding=0, bias=True))]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [spectral_norm(conv3x3(inplanes, outplanes))]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


# set norm_layer is none
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [spectral_norm(conv3x3(inplanes, inplanes))]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        if n_blocks == 6:
            max_ndf = n_blocks
        else:
            max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output




class RGBBlock(nn.Module):
    def __init__(self, nz, input_channel, upsample):
        super(RGBBlock, self).__init__()
        self.to_style = LayerEpilogue(input_channel, nz)
        self.conv = spectral_norm(nn.Conv2d(input_channel, 3, 3, 1, 1))
        # no need for activation function
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, pre_rgb, z):
        x = self.to_style(x, z)
        x = self.conv(x)
        if pre_rgb is not None:
            x_out = x + pre_rgb
        else:
            x_out = x
        if self.upsample is not None:
            x_out = self.upsample(x_out)
        return x, x_out


class PGResnetBlock(nn.Module):
    def __init__(self, fin, fout, nz, upsample=True, upsample_rgb=True):
        super(PGResnetBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(fin, affine=False)
            self.to_style_s = LayerEpilogue(fin, nz)
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        fmiddle = min(fin, fout)

        self.to_style_0 = LayerEpilogue(fin, nz)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, 3, 1, 1))
        self.norm_0 = nn.InstanceNorm2d(fin, affine=False)

        self.to_style_1 = LayerEpilogue(fmiddle, nz)
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, 3, 1, 1))
        self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=False)

        self.actv = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.to_rgb = RGBBlock(nz, fout, upsample_rgb)
    
    def forward(self, x, pre_rgb, z, edge=None):
        if self.upsample is not None:
            x = self.upsample(x)

        if edge is not None:
            edge = F.interpolate(edge, size=x.size()[2:], mode='bilinear')
            x = torch.cat([x, edge], 1)
        x1 = self.to_style_0(self.norm_0(x), z)
        x1 = self.conv_0(self.actv(x1))

        x2 = self.to_style_1(self.norm_1(x1), z)
        x2 = self.conv_1(self.actv(x2))

        if self.learned_shortcut:
            x_s = self.to_style_s(self.norm_s(x), z)
            x_s = self.conv_s(self.actv(x_s))
            x2 = x2 + x_s

        rgb_this_block, rgb = self.to_rgb(x2, pre_rgb, z)
        return x2, rgb, rgb_this_block


"Define progressive generator, add edge to each block, with to_RGB, to_style each block"


class Progressive_generator(nn.Module):
    def __init__(self, opt):
        super(Progressive_generator, self).__init__()
        nf = opt.ngf
        self.sw = opt.fineSize // (2 ** 7)
        self.sh = self.sw
        # upsample in each block *7
        self.fc = nn.Conv2d(opt.input_nc, 16 * nf, 3, padding=1)
        self.head_0 = PGResnetBlock(16 * nf + 1, 16 * nf, opt.nz)
        self.G_middle_0 = PGResnetBlock(16 * nf + 1, 16 * nf, opt.nz)
        self.G_middle_1 = PGResnetBlock(16 * nf + 1, 16 * nf, opt.nz)
        self.up_0 = PGResnetBlock(16 * nf + 1, 8 * nf, opt.nz)
        self.up_1 = PGResnetBlock(8 * nf + 1, 4 * nf, opt.nz)
        self.up_2 = PGResnetBlock(4 * nf + 1, 2 * nf, opt.nz)
        self.up_3 = PGResnetBlock(2 * nf + 1, 1 * nf, opt.nz, upsample_rgb=False)

        final_nc = nf

        self.visualize = opt.visualize
        # if opt.num_upsampling_layers == 'most':
        #     self.sw=opt.fine_size // (2**7)
        #     self.sh=self.sw
        #     self.up_4 = PGResnetBlock(1 * nf, nf // 2, opt)
        #     final_nc = nf // 2
        # final conv don't take edge to concat
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

    def forward(self, edge, latent_z):
        x = F.interpolate(edge, size=(self.sh, self.sw), mode='bilinear')
        x = self.fc(x)
        rgb = None
        if not self.visualize:
            x, rgb, _ = self.head_0(x, rgb, latent_z, edge)
            x, rgb, _ = self.G_middle_0(x, rgb, latent_z, edge)
            x, rgb, _ = self.G_middle_1(x, rgb, latent_z, edge)
            x, rgb, _ = self.up_0(x, rgb, latent_z, edge)
            x, rgb, _ = self.up_1(x, rgb, latent_z, edge)
            x, rgb, _ = self.up_2(x, rgb, latent_z, edge)
            _, rgb, _ = self.up_3(x, rgb, latent_z, edge)
            out = F.tanh(rgb)
            return out
        else:
            x, rgb, rgb_0 = self.head_0(x, rgb, latent_z, edge)
            x, rgb, rgb_1 = self.G_middle_0(x, rgb, latent_z, edge)
            x, rgb, rgb_2 = self.G_middle_1(x, rgb, latent_z, edge)
            x, rgb, rgb_3 = self.up_0(x, rgb, latent_z, edge)
            x, rgb, rgb_4 = self.up_1(x, rgb, latent_z, edge)
            x, rgb, rgb_5 = self.up_2(x, rgb, latent_z, edge)
            _, rgb, rgb_6 = self.up_3(x, rgb, latent_z, edge)
            rgb_final = F.tanh(rgb)
            return rgb_0, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, rgb_final



class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale=False,
                 use_noise=False,
                 ):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        else:
            self.noise = None

        self.linear = nn.Linear(dlatent_size, channels * 2)
        # self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)

    def forward(self, x, z):
        style = self.linear(z)  # style => [batch_size, n_channels*2]
        style = F.leaky_relu(style, 0.2, inplace=True)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]

        return x


