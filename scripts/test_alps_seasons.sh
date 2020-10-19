python ./test.py --model zvae_wgan --netG progressive_256 --where_add AdaIN --nz 64 --dataroot E://pku/dataset/alps_season/256/ --results_dir ./results/alps_season/ --checkpoints_dir ./checkpoints/alps_seasons/ --epoch latest --name alps_seasons_progressive_64 --direction AtoB --loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --phase val

