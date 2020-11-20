import argparse
import os
import sys

import torch

import data
import models
from util import util


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--g_input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--nz', type=int, default=8, help='#latent vector')
        parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        parser.add_argument('--name', type=str, default='',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop|crop|scaledCrop|scaleWidth|scaleWidth_and_crop|scaleWidth_and_scaledCrop|scaleHeight|scaleHeight_and_crop] etc')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single,labeled')
        parser.add_argument('--model', type=str, default='bicycle_gan',
                            help='chooses which model to use. bicycle,, ...')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--vggLoss', action='store_true', help='whether apply the vgg loss : relu2-2+relu3_3')
        # models
        parser.add_argument('--num_Ds', type=int, default=2, help='number of Discrminators')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='dcgan|lsgan|wgan-gp|hinge')
        parser.add_argument('--netD', type=str, default='basic_256_multi',
                            help='selects model to use for netD, basic_256_multi for normal, basic_256_multi_class for adding class label')
        parser.add_argument('--netD2', type=str, default='basic_256_multi', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='progressive_256', help='selects model to use for netG')
        parser.add_argument('--netE', type=str, default='resnet_256', help='selects model to use for netE')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--use_res', action='store_true', help='whether use resblock in unet')

        # extra parameters
        parser.add_argument('--where_add', type=str, default='AdaIN',
                            help='input|all|middle|AdaIN; where/HOW to add z in the network G')
        parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--center_crop', action='store_true', help='if apply for center cropping for the test')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        # for semantic map
        parser.add_argument('--label_nc', type=int, default=182,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dontcare_label.')  # coco 182; ADE20K 150

        # for edge detection (DexiNet)
        parser.add_argument('--edge_mode', type=str, default="average", help="which edge mode to use [average|fuse]")
        parser.add_argument('--use_edgeNet', type=bool, default=True, help='Whether use networks for edge detection')
        parser.add_argument('--DexiNet_cp', type=str, default='../edge_detection/DexiNed/DexiNed-Pytorch/checkpoints/24/24_model.pth',
                            help='checkpoint path for edge detection network (DexiNed).')

        # train options
        parser.add_argument('--num_val', type=int, default=1000, help='how many validation images to run')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='whether use TTUR training scheme')
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        # lambda parameters
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam,0.0002 by default')
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|,defaut 10,,40')
        parser.add_argument('--lambda_GAN', type=float, default=1.0,
                            help='weight on D loss. D(G(A, E(B))), default 1,,2.0')
        # parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
        parser.add_argument('--lambda_z', type=float, default=1, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        parser.add_argument('--lambda_ssim', type=float, default=0.25, help='weight for ssim loss,default 0.1')
        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')
        parser.add_argument('--lambda_feature_loss', type=float, default=0.3, help='weight for feature loss')
        # visualize
        parser.add_argument('--visualize',action='store_true',default=False,help='whether to visualize each rgb pic of progressive generator ')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
