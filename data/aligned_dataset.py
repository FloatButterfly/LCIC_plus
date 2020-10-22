import os.path
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
from edge_detection.DexiNed.DexiNed_Pytorch.model import DexiNet

import pdb

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_AB = "/backup1/home/zhangyuefeng/programs/test"
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
        
            # TODO: why (0.5, 0.5, 0.5) here? mean?
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

            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

        else:
            AB_path = self.AB_paths[index]
            file_name = [os.path.basename(AB_path)]

            # TODO: add edge detection algorithm on the fly
            # A = Image.open(AB_path).convert('RGB')
            A = cv2.imread(AB_path, cv2.IMREAD_COLOR)
            checkpoint_path = "/backup1/home/zhangyuefeng/programs/edge_detection/DexiNed/DexiNed_Pytorch/checkpoints/24/24_model.pth"
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint filte note found: {checkpoint_path}")
            print(f"Restoring weights from: {checkpoint_path}")

            device = torch.device('cuda' if len(self.opt.gpu_ids) > 0 else 'cpu')
            edge_model = DexiNet().to(device)

            edge_model.load_state_dict(torch.load(checkpoint_path,
                                            map_location=device))
            edge_model.eval()

            # image preprocess: size transforms
            A_shape = [A.shape[0], A.shape[1]]
            if A.shape[0] < 512 or A.shape[1] < 512:
                A = cv2.resize(A, (512, 512))
                print("actual size: {}, target size: {}".format(A_shape, (512, 512)))
            elif A.shape[0] % 16 != 0 or A.shape[1] % 16 != 0:
                A_height = ((A.shape[0] // 16) + 1) * 16 if A_shape[0] % 16 != 0 else A.shape[0]
                A_width = ((A.shape[1] // 16) + 1) * 16 if A_shape[1] % 16 != 0 else A.shape[1]
                A = cv2.resize(A, (A_width, A_height))
                print("actual size: {}, target size: {}".format(A_shape, (A_width, A_height)))
            
            mean_pixel_values = [103.939,116.779,123.68]
            print("mean pixel values: {}".format(mean_pixel_values))

            A = np.array(A, dtype=np.float32)
            A -= mean_pixel_values
            A = A.transpose((2,0,1))
            A = torch.from_numpy(A.copy()).float()

            with torch.no_grad():
                # A = transforms.ToTensor()(A)
                A = A.to(device)
                output = edge_model(torch.unsqueeze(A, dim=0))
            pdb.set_trace()
            fuse, _ = self._save_output(output, file_name, A_shape, save=True)

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
            
            return {'A': A, 'B': B}
            # return {'A': A, 'B': B,
            #         'A_paths': AB_path, 'B_paths': AB_path}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def _save_output(self, tensor, file_names, img_shape=None, output_dir=".", save=False):
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
        print("image_shape:{}".format(image_shape))
        
        pdb.set_trace()
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

            if save:
                cv2.imwrite(os.path.join(output_dir, "fuse_" + file_name), fuse)
                cv2.imwrite(os.path.join(output_dir, "avg_" + file_name), average)
            
            idx += 1

        return fuse, average


if __name__ == '__main__':
    print("Running")