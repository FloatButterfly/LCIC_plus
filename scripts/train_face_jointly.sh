# train jointly
# python train.py --conditional_D --netG progressive_256 --netD basic_256_multi --gan_mode ls --vggLoss --where_add AdaIN --nz 64 --model zvae_wgan \
# --dataroot E://pku/dataset/alps_season/256/ --phase train --name alps_seasons_progressive_64 --batch_size 4 --display_id 10 --direction AtoB \
# --checkpoints_dir ./checkpoints/alps_seasons/ --loadSize 256 --fineSize 256 --input_nc 1 --niter 60 --niter_decay 60 --lr 0.00005 --continue_train

CUDA_VISIBLE_DEVICES=2 python ./train.py \
--conditional_D \
--netD basic_256_multi \
--netG progressive_256 \
--gan_mode ls \
--model zvae_wgan \
--where_add AdaIN \
--nz 64 \
--dataroot ../../dataset/CelebAMask-HQ/CelebA-HQ-img \
--checkpoints_dir ./checkpoints/ \
--epoch latest \
--name face_progressive_64_jointly \
--DexiNet_cp /media/sdc/yuefeng/programs/edge_detection/DexiNed/DexiNed-Pytorch/checkpoints/256/24/24_model.pth \
--batch_size 2 \
--display_id 10 \
--direction AtoB \
--loadSize 256 \
--fineSize 256 \
--input_nc 1 \
--center_crop \
--no_flip \
--niter_decay 60 \
--lr 0.00005 \
--phase train \
--continue_train \
--epoch_count 2 \
--num_val 1000
# --vggLoss \
