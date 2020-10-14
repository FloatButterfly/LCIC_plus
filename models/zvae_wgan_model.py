import torch

from . import networks
from .base_model import BaseModel
import util.util as util
from numpy import int16, int32


# from models import EMA

class zVaeWGANModel(BaseModel):
    def name(self):
        return 'zVaeWGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        # if opt.isTrain:
        #     assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # self.ema_updater = EMA(0.99)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.tensor_name = ['z_encoded']
        if opt.gan_mode == 'wgan-gp':
            self.loss_names = ['G_GAN', 'D', 'G_L1', 'z_L1', 'kl', 'feature', 'D_epsilon', 'D_fake', 'D_real', 'D_grad']
        else:
            self.loss_names = ['G_GAN', 'D', 'G_L1', 'z_L1', 'kl', 'feature', 'ssim', 'D_fake', 'D_real']
        # self.loss_names = ['grad_all']
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_encoded']
        self.logname = 'testZ_log.txt'
        # specify the models you want to save to the disk. The program will call base_model.save_networks and
        # base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_E = opt.isTrain or not opt.no_encode
        self.use_feature_loss = opt.vggLoss is True
        use_vae = True
        self.gan_mode = opt.gan_mode

        self.model_names = ['G']
        self.netG = networks.define_G(opt, gpu_ids=self.gpu_ids)

        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc

        use_sigmoid = self.gan_mode == 'dcgan'

        if self.use_feature_loss:
            self.vgg = networks.define_Feature_Net(False, 'vgg16', self.gpu_ids)

        if use_D:
            self.model_names += ['D']
            if self.gan_mode == 'wgan-gp':
                norm = "instance"
            else:
                norm = opt.norm
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                          gpu_ids=self.gpu_ids)
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.ndf, netE=opt.netE, output_nc=opt.nz, norm=opt.norm,
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        if opt.isTrain:
            if self.gan_mode == 'wgan-gp':
                self.criterionGAN = networks.WGANGP().to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionZ = torch.nn.L1Loss()
            self.criterion_SSIM = networks.SSIM_loss(size_average=True, data_range=1.0)

            # initialize optimizers
            self.optimizers = []
            if opt.no_TTUR:
                G_lr, E_lr, D_lr = opt.lr, opt.lr, opt.lr / 2
            else:
                G_lr, E_lr, D_lr = opt.lr / 2, opt.lr / 2, opt.lr * 2

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=G_lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=E_lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_E)
            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=D_lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # import pdb
        # pdb.set_trace()
        self.image_paths = input['A_paths' if AtoB else 'B_paths'][0].split('\\')[-1].split('.')[0]

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def z_encode(self):
        z, logvar = self.netE(self.real_B)
        self.z_encoded = z
        return self.z_encoded

    def test_interpolate(self, z0=None, encode=True):
        with torch.no_grad():
            if encode:
                z_A, logvar = self.netE(self.real_A)
                z_B, _ = self.netE(self.real_B)
                return z_A, z_B
            else:
                z0 = z0.to(self.device)
                self.fake_B = self.netG(self.real_A, z0)
                return self.real_A, self.fake_B, self.real_B

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, logvar = self.netE(self.real_B)
                # std = logvar.mul(0.5).exp_()
                # eps = self.get_z_random(std.size(0), std.size(1))
                # z0 = eps.mul(std).add_(mu)
                self.z_encoded = z0
                self.logvar = logvar
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            z0 = z0.to(self.device)
            if not self.opt.visualize:
                self.fake_B = self.netG(self.real_A, z0)
                return self.real_A, self.fake_B, self.real_B
            else:
                rgb_0, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, rgb_final = self.netG(self.real_A, z0)
                return self.real_A, self.real_B, rgb_0, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, rgb_final
            # print(z0)
            # return self.z_encoded, self.real_A, self.fake_B, self.real_B, self.logvar

    def test_encode(self, z0=None, encode=False, qp=-1, z_log_dir=None):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, logvar = self.netE(self.real_B)
                # std = logvar.mul(0.5).exp_()
                # eps = self.get_z_random(std.size(0), std.size(1))
                # z0 = eps.mul(std).add_(mu)
                self.z_encoded = z0
                self.logvar = logvar
                if qp > 0:
                    z = z0 * (2 ** (10 - (qp - 4) / 6))
                    z = z.floor()
                    if z.min() < -2e16 or z.max() > 2e16:
                        print("warning,out of int16")
                        z = torch.clamp(z, -2e16, 2e16)
                    z = z.to(torch.int16)
                    # import pdb
                    # pdb.set_trace()
                    xx = z.cpu().detach().numpy().astype(int32)
                    if z_log_dir is None:
                        z_log_dir = "z_%s_q.bin" % self.image_paths
                    with open(z_log_dir, "a") as log_file:
                        xx.tofile(log_file)
                        # np.savetxt(log_file, xx, fmt='%d')
                    uz = z.to(torch.float32)
                    uz = uz * (2 ** ((qp - 4) / 6 - 10))
                    error = (uz - z0).abs().mean()
                    print("quantization error:", error)
                    z0 = uz.clone()
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)

            z0 = z0.to(self.device)
            self.fake_B = self.netG(self.real_A, z0)
            print(z0)
            # return self.z_encoded, self.real_A, self.fake_B, self.real_B, self.logvar
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        if self.opt.conditional_D:  # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict with fake_B_encoded
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_encoded)
        # if self.opt.lambda_z > 0.0:
        #     self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate
        # feature loss
        if self.use_feature_loss:
            fake_F2_denorm = util.de_normalise(self.fake_B_encoded)
            real_F2_denorm = util.de_normalise(self.real_B_encoded)
            fake_F2_norm = util.normalise_batch(fake_F2_denorm)
            real_F2_norm = util.normalise_batch(real_F2_denorm)
            self.features_fake_F2 = self.vgg(fake_F2_norm)
            self.features_real_F2 = self.vgg(real_F2_norm)

    # ***************change point******************
    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake, fake_mean = netD(fake.detach())
        # real
        pred_real, real_mean = netD(real)
        if self.gan_mode == 'wgan-gp':
            # loss_D_fake = self.criterionGAN(pred_fake, False).item()
            # loss_D_real = self.criterionGAN(pred_real, True).item()
            self.loss_D_fake = fake_mean
            self.loss_D_real = -real_mean
            self.loss_D_grad = networks.WGANGPGradientPenalty(real, fake.detach(), netD, weight=10, backward=True)
            # small weight to keep the discriminator output from drifting too far away from zero. To be precise,
            # we set L0 = L + driftEx2Pr [D(x)2], where drift = 0.001.
            self.loss_D_epsilon = (real_mean ** 2) * 0.001
            loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_grad + self.loss_D_epsilon
        else:
            self.loss_D_fake, = self.criterionGAN(pred_fake, False, for_discriminator=True)
            self.loss_D_real, = self.criterionGAN(pred_real, True, for_discriminator=True)
            # Combined loss
            loss_D = self.loss_D_fake + self.loss_D_real
        loss_D.backward()

        if self.gan_mode == 'wgan-gp':
            return loss_D, [self.loss_D_fake, self.loss_D_real, self.loss_D_grad, self.loss_D_epsilon]
        return loss_D, [self.loss_D_fake, self.loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake, mean = netD(fake)
            if self.gan_mode == 'wgan-gp':
                # loss_G_GAN = self.criterionGAN(pred_fake, True).item()
                # two discrinator, for convenience, calcu mean output
                loss_G_GAN = -mean
            else:
                loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)

        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        if self.opt.lambda_ssim > 0.0:
            self.loss_ssim = self.criterion_SSIM(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_ssim
        else:
            self.loss_ssim = 0.0

        # vgg loss
        if self.use_feature_loss:
            self.loss_feature = (self.criterionL2(self.features_fake_F2.relu2_2, self.features_real_F2.relu2_2) +
                                 self.criterionL2(self.features_fake_F2.relu3_3,
                                                  self.features_real_F2.relu3_3)) * self.opt.lambda_feature_loss
        else:
            self.loss_feature = 0.0

        self.share_loss = self.loss_G_GAN + self.loss_G_L1 + self.loss_ssim + self.loss_feature

        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_encoded)) * self.opt.lambda_z
        else:
            self.loss_z_L1 = 0.0

        self.loss_G = self.share_loss + self.loss_z_L1

        self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        self.loss_E = self.share_loss + self.loss_kl

        self.optimizer_E.zero_grad()
        self.loss_E.backward()
        self.optimizer_E.step()
        # vgg loss
        # return self.loss_G
        # self.loss_G = self.loss_feature
        # self.loss_G.backward()
        # gl = []
        # for parameter in self.netG.named_parameters():
        #     x = parameter[1].grad
        #     if x is not None:
        #         gl.append(x.abs().mean().item())
        # else:
        #     print(parameter[0], parameter[1], x)
        # self.loss_grad_all = torch.tensor(gl).mean().item()

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            self.optimizer_D.step()

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.backward_EG()
        # update G
        # self.optimizer_G.zero_grad()
        # share_loss = self.shareloss_EG()
        # self.backward_G(share_loss)
        # self.optimizer_G.step()
        # # update E
        # self.optimizer_E.zero_grad()
        # self.backward_E(share_loss)
        # self.optimizer_E.step()
        # #

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()
