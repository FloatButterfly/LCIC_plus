import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_img_params(opt, size):
    w, h = size
    new_h, new_w = h, w
    if 'resize' in opt.resize_or_crop:  # resize image to be loadSize x loadSize
        new_h = new_w = opt.loadSize
    elif 'scaleWidth' in opt.resize_or_crop:  # scale image width to be loadSize
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w
    elif 'scaleHeight' in opt.resize_or_crop:  # scale image height to be loadSize
        new_h = opt.loadSize
        new_w = opt.loadSize * w // h
    elif 'randomScaleWidth' in opt.resize_or_crop:  # randomly scale image width to be somewhere between loadSize and
        #  fineSize
        new_w = random.randint(opt.fineSize, opt.loadSize + 1)
        new_h = new_w * h // w
    elif 'randomScaleHeight' in opt.resize_or_crop:  # randomly scale image height to be somewhere between loadSize
        # and fineSize
        new_h = random.randint(opt.fineSize, opt.loadSize + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4

    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        if 'crop' in opt.resize_or_crop:  # crop patches of size fineSize x fineSize
            crop_w = crop_h = opt.fineSize
        else:
            if 'Width' in opt.resize_or_crop:  # crop patches of width fineSize
                crop_w = opt.fineSize
                crop_h = opt.fineSize * h // w
            else:  # crop patches of height fineSize
                crop_h = opt.fineSize
                crop_w = opt.fineSize * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)
        x_span = (new_w - crop_w) // 2
        crop_x = np.maximum(0, np.minimum(x_span * 2, int(np.random.randn() * x_span / 3 + x_span)))
        crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
        # crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
        # crop_y = random.randint(0, np.maximum(0, new_h - crop_h))
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)

    flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
    return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    ### resize input image
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    ### crop patches from image
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))

        ### random flip
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_single(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop_and_resize':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.RandomCrop(opt.fineSize))
        transform_list.append(transforms.Scale(osize, Image.ANTIALIAS))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def toTensor_normalize():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)


def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos
    if (ow > tw or oh > th):
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
