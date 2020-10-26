CUDA_VISIBLE_DEVICES=3 python ./test.py --model zvae_wgan --netG progressive_256 --where_add AdaIN --nz 64 --dataroot /media/sdc/yuefeng/dataset/seasons/ --results_dir ./results/alps_season/ --checkpoints_dir ./checkpoints/ --epoch latest --name alps_seasons_progressive_64 --direction AtoB --loadSize 256 --fineSize 256 --input_nc 1 --num_test 10 --center_crop --no_flip --phase test

