import os
import time
import torch
from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util.visualizer import Visualizer
from pytorch_msssim import ssim
import numpy as np


def test(model, val_dataset):
    model.eval()
    cur_score = 0.0
    with torch.no_grad():
        for i, images in enumerate(val_dataset):
            model.set_input(images)
            real_A, fake_B, real_B = model.test(encode=True)
            cur_score += ssim(fake_B, real_B, val_range=1.0)
        cur_score /= len(val_dataset)
    return cur_score


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = TestOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    val_opt.phase = 'test'
    val_opt.batch_size = 1
    val_opt.num_threads = 1
    val_opt.serial_batches = True  # no shuffle
    val_dataloader = CreateDataLoader(val_opt)
    val_dataset = val_dataloader.load_data()
    print('val images = %d' % len(val_dataloader))

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    best_score = 0
    log_dir = os.path.join(opt.checkpoints_dir, "val_ssim.txt")

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        t_data = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            if not model.is_train():
                continue
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                tensors = model.get_tensor_encoded()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_Tensor_encoded(epoch, epoch_iter, tensors)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
                # ========== Validation ========
                print('Validation with ssim..')
                cur_score = test(model, val_dataset)
                print(" ssim = %.6f" % cur_score)
                with open(log_dir, 'a') as log_file:
                    log_file.write("ssim value of epoch %d is %.6f \n" % (epoch, cur_score))
                if cur_score > best_score:
                    best_score = cur_score
                    model.save_networks('best')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
