import os
import pdb
import time
from itertools import islice

import numpy as np

from data import CreateDataLoader
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1  # test code only supports num_threads=1
    opt.batch_size = 1  # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    # create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print('Loading model %s' % opt.model)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

    # z_log_dir_q = os.path.join(opt.results_dir, 'z_vector_q.bin')
    # z_log_dir = os.path.join(opt.results_dir, 'z_vector.txt')
    # std_log_dir = os.path.join(opt.results_dir, 'z_std.txt')
    # logvar_log_dir = os.path.join(opt.results_dir, 'z_logvar.txt')

    # test stage
    for i, data in enumerate(dataset):
        start_time = time.time()
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test if opt.num_test != -1 else len(dataset)))

        A_path = data['A_paths'][0]
        z_log_q_path = os.path.join(A_path.split('/')[-2], 
                        os.path.basename(A_path).split('.')[0] + "_z_log_q.bin")
        z_log_q_path = os.path.join(opt.results_dir, z_log_q_path)

        encode = True
        if opt.model == "zvae_gan":
            real_A, fake_B, real_B = model.test(encode=encode, qp=opt.qp, z_log_path=z_log_q_path)
        elif opt.model == "zvae_wgan":
            real_A, fake_B, real_B = model.test_encode(encode=encode, qp=opt.qp, z_log_path=z_log_q_path)
            # real_A, fake_B, real_B = model.test(encode=encode)
        else:
            real_A, fake_B, real_B = model.test(encode=encode)

        # z = model.z_encode()
        # with open(z_log_dir, "a") as log_file:
        #     np.savetxt(log_file, z.cpu().detach().numpy())
        # with open(std_log_dir, "a") as log_file:
        #     np.savetxt(log_file, std.cpu().detach().numpy())
        # with open(logvar_log_dir, "a") as log_file:
        #     np.savetxt(log_file, logvar.cpu().detach().numpy())
        images = [real_A, real_B, fake_B]
        names = ['input', 'ground truth', 'encoded']

        img_path = os.path.join(opt.results_dir, A_path.split('/')[-2], os.path.basename(A_path))
        # img_path = 'input_%3.3d' % i
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

        print(f"time spent: {time.time() - start_time}s")

        if opt.num_test == -1:
            continue
        if i + 1 == opt.num_test:
            break

    webpage.save()

