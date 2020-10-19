from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        # parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_port', type=int, default=9988, help='visdom display port')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--update_html_freq', type=int, default=4000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        # parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # parser.add_argument('--epoch_count', type=int, default=1,
        #                     help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        # parser.add_argument('--niter_decay', type=int, default=100,
        #                     help='# of iter to linearly decay learning rate to zero')

        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # learning rate

        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=100,
                            help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
