from models.edge_model import edgeModel
import os.path
import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

import sys
sys.path.insert(0, "..")
# from edge_detection.DexiNed.DexiNed_Pytorch.model import DexiNet
from models import edge_model

import pdb

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert (opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        # use combined input (RGB image and its corresponding edge image)
        use_combined = False

        if use_combined:
            AB_path = self.AB_paths[index]
            AB = Image.open(AB_path).convert('RGB')
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)
            w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

            A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        
            print("A shape before norm: {}".format(A.shape))
            print("B shape before norm: {}".format(B.shape))

            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

            if self.opt.direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)

            # channel number ,1 gray
            if input_nc == 1:  # RGB to gray， 0.2989 R + 0.5870 G + 0.1140 B 
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)
            
            # print("A shape final:", A.shape)
            # print("B shape final:", B.shape)

            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

        else:
            AB_path = self.AB_paths[index]
            file_name = [os.path.basename(AB_path)]

            # A: edge image, B: RGB groudtruth
            B = cv2.imread(AB_path, cv2.IMREAD_COLOR)       # BGR
            B = cv2.resize(B, (self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_CUBIC)
            h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
            w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
            B = B[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

            # use DexiNet to extract edge feature
            checkpoint_path = self.opt.DexiNet_cp
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint filte note found: {checkpoint_path}")
            # print(f"Restoring DexiNed weights from: {checkpoint_path}")

            device = torch.device('cuda' if len(self.opt.gpu_ids) > 0 else 'cpu')
            edge_model = edgeModel().to(device)

            edge_model.load_state_dict(torch.load(checkpoint_path,
                                            map_location=device))
            edge_model.eval()
            
            # image preprocess: size transforms
            B_shape = [B.shape[0], B.shape[1]]
            # if B.shape[0] < 512 or B.shape[1] < 512:
            #     B = cv2.resize(B, (512, 512))
            #     print("actual size: {}, target size: {}".format(B_shape, (512, 512)))
            # elif B.shape[0] % 16 != 0 or B.shape[1] % 16 != 0:
            #     B_height = ((B.shape[0] // 16) + 1) * 16 if B_shape[0] % 16 != 0 else B.shape[0]
            #     B_width = ((B.shape[1] // 16) + 1) * 16 if B_shape[1] % 16 != 0 else B.shape[1]
            #     B = cv2.resize(B, (B_width, B_height))
            #     print("actual size: {}, target size: {}".format(B_shape, (B_width, B_height)))
            
            mean_pixel_values = [103.939,116.779,123.68]
            # print("mean pixel values: {}".format(mean_pixel_values))

            B_input = np.array(B, dtype=np.float32).copy()
            B_input -= mean_pixel_values
            B_input = B_input.transpose((2,0,1))        # RGB
            B_input = torch.from_numpy(B_input.copy()).float()

            with torch.no_grad():
                B_input = B_input.to(device)
                B_output = edge_model(torch.unsqueeze(B_input, dim=0))

            fuse, avg = self._save_output(B_output, file_name, B_shape, save=False)
            
            if self.opt.edge_mode == 'average':
                A = transforms.ToTensor()(avg.copy()).to(device)
            elif self.opt.edge_mode == 'fuse':
                A = transforms.ToTensor()(fuse.copy()).to(device)
            else:
                raise ValueError('Unknown edge mode: {}'.format(self.opt.edge_mode))

            B = B[:, :, [2, 1, 0]]          # to RGB
            B = transforms.ToTensor()(B).to(device)

            # print("A shape: {}".format(A.shape))
            # print("B shape: {}".format(B.shape))

            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

            # im1 = transforms.ToPILImage()(B.cpu()).convert("RGB")
            # im1.save("out_B.jpg")
            # im2 = transforms.ToPILImage()(A.cpu()).convert("RGB")
            # im2.save("out_A.jpg")

            if self.opt.direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)

            # channel number ,1 gray
            if input_nc == 1:  # RGB to gray， 0.2989 R + 0.5870 G + 0.1140 B 
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)
            
            return {'A': A, 'B': B, 'A_paths': AB_path, "B_paths": AB_path}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def _save_output(self, tensor, file_names, img_shape=None, save=False):
        if self.opt.phase == 'train':
            output_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'edge_maps')
        elif self.opt.phase == 'test':
            output_dir = os.path.join(self.opt.results_dir, 'edge_maps')
        
        # 255.0 * (1.0 - em_a)
        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)

        image_shape = [x for x in img_shape]
        image_shape = [np.array([x]) for x in image_shape]
        # (H, W) -> (W, H)
        image_shape =[[y, x] for x, y in zip(image_shape[0], image_shape[1])]
        # print("image_shape:{}".format(image_shape))
        
        idx = 0
        for i_shape, file_name in zip(image_shape, file_names):
            tmp = tensor[:, idx, ...]
            tmp = np.squeeze(tmp)

            # Iterate our all 7 NN outputs for a particular image
            preds = []
            for i in range(tmp.shape[0]):
                tmp_img = tmp[i]
                tmp_img[tmp_img < 0.0] = 0.0
                tmp_img = 255.0 * (1.0 - tmp_img)

                # Resize prediction to match input image size
                if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                    tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))

                preds.append(tmp_img)
                if i == 6:
                    fuse = tmp_img

            fuse = fuse.astype(np.uint8)
            
            # Get the mean prediction of all the 7 outputs  
            average = np.array(preds, dtype=np.float32)
            average = np.uint8(np.mean(average, axis=0))
            
            # print("average shape:", average.shape)
            if save:
                fuse_dir = os.path.join(output_dir, 'fuse')
                if not os.path.exists(fuse_dir):
                    os.makedirs(fuse_dir)
                cv2.imwrite(os.path.join(fuse_dir, "fuse_" + file_name), fuse)
                print("save fused image at: ", os.path.join(fuse_dir, "fuse_" + file_name))

                avg_dir = os.path.join(output_dir, 'average')
                if not os.path.exists(avg_dir):
                    os.makedirs(avg_dir)
                cv2.imwrite(os.path.join(avg_dir, "avg_" + file_name), average)
                print("save average image at: ", os.path.join(avg_dir, "avg_" + file_name))

            idx += 1

        # return rgb image
        fuse = np.tile(fuse[..., np.newaxis], (1, 1, 3))
        fuse = fuse[...,::-1]
        average = np.tile(average[..., np.newaxis], (1, 1, 3))
        average = average[...,::-1]

        # print("here fuse shape:", fuse.shape)
        # print("here average shape:", average.shape)

        return fuse, average


if __name__ == '__main__':
    print("Running")