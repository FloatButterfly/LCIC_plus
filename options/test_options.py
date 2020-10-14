from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
        parser.add_argument('--test_psnr', action='store_true', help='test the quality of encoded image by psnr')
        parser.add_argument('--test_ssim', action='store_true', help='test the quality of encoded image by ssim')
        parser.add_argument('--test_msssim', action='store_true', help='test the quality of encoded image by ms-ssim')
        parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        parser.add_argument('--qp', type=int, default=0, help='quantization params: <0 means no quantization')
        self.isTrain = False
        parser.set_defaults(batch_size=1)
        parser.set_defaults(num_threads=1)
        return parser
