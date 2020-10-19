#
#
python train.py --conditional_D --netG progressive_256 --netD basic_256_multi --gan_mode ls --vggLoss --where_add AdaIN --nz 64 --model zvae_wgan \
--dataroot E://pku/dataset/alps_season/256/ --phase train --name alps_seasons_progressive_64 --batch_size 4 --display_id 10 --direction AtoB \
--checkpoints_dir ./checkpoints/alps_seasons/ --loadSize 256 --fineSize 256 --input_nc 1 --niter 60 --niter_decay 60 --lr 0.00005 --continue_train