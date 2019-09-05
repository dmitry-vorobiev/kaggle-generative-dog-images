import math
import os
import cv2
import numpy as np
from functools import partial
from numba import jit
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensor
import xml.etree.ElementTree as ET


IMG_SIZE = 64
IMG_SIZE_2 = IMG_SIZE * 2
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MyImg:
    def __init__(self, img, tfm):
        self.px = np.array(img)
        self.tfm = tfm

    @property
    def size(self):
        h, w, _ = self.px.shape
        return min(w, h)


def pad(img, padding_mode='reflect'):
    p = math.ceil((max(img.size) - min(img.size)) / 2)
    p_horr = p if img.width < img.height else 0
    p_vert = p if img.height < img.width else 0
    img = T.Pad((p_horr, p_vert), padding_mode=padding_mode)(img)
    if img.width != img.height:
        s = min(img.size)
        img = img.crop((0, 0, s, s))
    return img


def take_top(img):
    size = min(img.size)
    bbox = (0, 0, size, size)
    return img.crop(bbox)


def take_diagonal(img):
    w, h = img.size
    size = min(w, h)
    bbox_l = (0, 0, size, size)
    bbox_r = (w - size, h - size, w, h)
    return [img.crop(bbox_l), img.crop(bbox_r)]


resize = T.Resize(IMG_SIZE, interpolation=Image.LANCZOS)
resize2x = T.Resize(IMG_SIZE_2, interpolation=Image.LANCZOS)

center_crop = T.Compose([resize, T.CenterCrop(IMG_SIZE)])
center_crop2x = T.Compose([resize2x, T.CenterCrop(IMG_SIZE_2)])

top_crop = T.Compose([T.Lambda(take_top), resize])
top_crop2x = T.Compose([T.Lambda(take_top), resize2x])

two_crops = T.Compose([resize, T.Lambda(take_diagonal)])
two_crops2x = T.Compose([resize2x, T.Lambda(take_diagonal)])

pad_only = T.Compose([T.Lambda(pad), resize])
pad_only2x = T.Compose([T.Lambda(pad), resize2x])


@jit(nopython=True)
def calc_one_axis(clow, chigh, pad, cmax):
    clow = max(0, clow - pad)
    chigh = min(cmax, chigh + pad)
    return clow, chigh, chigh - clow


def calc_bbox(obj, img_w, img_h, zoom=0.0, try_square=True):
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    # occasionally i get bboxes which exceed img size
    xmin, xmax, obj_w = calc_one_axis(xmin, xmax, 0, img_w)
    ymin, ymax, obj_h = calc_one_axis(ymin, ymax, 0, img_h)

    if zoom != 0.0:
        pad_w = obj_w * zoom / 2
        pad_h = obj_h * zoom / 2
        xmin, xmax, obj_w = calc_one_axis(xmin, xmax, pad_w, img_w)
        ymin, ymax, obj_h = calc_one_axis(ymin, ymax, pad_h, img_h)

    if try_square:
        # try pad both sides equaly
        if obj_w > obj_h:
            pad = (obj_w - obj_h) / 2
            ymin, ymax, obj_h = calc_one_axis(ymin, ymax, pad, img_h)
        elif obj_h > obj_w:
            pad = (obj_h - obj_w) / 2
            xmin, xmax, obj_w = calc_one_axis(xmin, xmax, pad, img_w)

        # if it's still not square, try pad where possible
        if obj_w > obj_h:
            pad = obj_w - obj_h
            ymin, ymax, obj_h = calc_one_axis(ymin, ymax, pad, img_h)
        elif obj_h > obj_w:
            pad = obj_h - obj_w
            xmin, xmax, obj_w = calc_one_axis(xmin, xmax, pad, img_w)

    return int(xmin), int(ymin), int(xmax), int(ymax)


@jit(nopython=True)
def bb2wh(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def make_x2res(img, bbox):
    if min(bb2wh(bbox)) < IMG_SIZE_2:
        return
    ar = img.width / img.height
    if ar == 1.0:
        tfm_img = resize2x(img)
    elif 1.0 < ar < 1.15:
        tfm_img = center_crop2x(img)
    elif 1.15 < ar < 1.25:
        tfm_img = pad_only2x(img)
    elif 1.25 < ar < 1.5:
        tfm_img = two_crops2x(img)
    elif 1.0 < 1 / ar < 1.6:
        tfm_img = top_crop2x(img)
    else:
        tfm_img = None
    return tfm_img


def add_sample(samples, label, tfm, imgs, labels):
    if not samples:
        return
    elif isinstance(samples, Image.Image):
        imgs.append(MyImg(samples, tfm))
        labels.append(label)
    elif isinstance(samples, list):
        imgs.extend([MyImg(s, tfm) for s in samples])
        labels.extend([label] * len(samples))
    else:
        assert False


def is_valid_file(x):
    return datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)


class DogsDataSet(datasets.vision.VisionDataset):
    def __init__(self, root, label_root, transforms, target_transform=None, max_samples=None):
        super().__init__(root, transform=None)
        assert isinstance(transforms, list) and len(transforms) == 3
        self.transforms = transforms
        self.target_transform = target_transform
        self.max_samples = max_samples
        self.classes = {}

        imgs, labels = self._load_subfolders_images(self.root, label_root)
        assert len(imgs) == len(labels)
        if len(imgs) == 0:
            raise RuntimeError(f'Found 0 files in subfolders of: {self.root}')
        self.imgs = imgs
        self.labels = labels

    def _create_or_get_class(self, name):
        try:
            label = self.classes[name]
        except KeyError:
            label = len(self.classes)
            self.classes[name] = label
        return label

    def _load_subfolders_images(self, root, label_root):
        light_zoom, medium_zoom = 0.08, 0.12
        n_pad, n_center, n_top, n_2crops, n_skip, n_dup, n_noop = 0, 0, 0, 0, 0, 0, 0
        imgs, labels, paths = [], [], []

        add_sample_ = partial(add_sample, imgs=imgs, labels=labels)

        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                paths.append(path)
        if self.max_samples:
            paths = paths[:self.max_samples]

        for path in paths:
            if not is_valid_file(path):
                continue
            img = datasets.folder.default_loader(path)
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(
                dirname for dirname in os.listdir(label_root) if dirname.startswith(annotation_basename.split('_')[0]))
            annotation_filename = os.path.join(label_root, annotation_dirname, annotation_basename)
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                name = o.find('name').text
                label = self._create_or_get_class(name)
                prev_bbox, tfm_imgs = None, None

                bbox = calc_bbox(o, img_w=img.width, img_h=img.height, zoom=light_zoom)
                obj_img = img.crop(bbox)
                add_sample_(make_x2res(obj_img, bbox), label, 2)

                bbox = calc_bbox(o, img_w=img.width, img_h=img.height)
                if min(bb2wh(bbox)) < IMG_SIZE:
                    # don't want pixel mess in gen imgs
                    n_skip += 1
                    continue
                obj_img = img.crop(bbox)
                ar = obj_img.width / obj_img.height
                if ar == 1.0:
                    tfm_imgs = [resize(obj_img)]
                    n_noop += 1
                elif 1.0 < ar < 1.3:
                    tfm_imgs = [center_crop(obj_img), pad_only(obj_img)]
                    n_center += 1
                    n_pad += 1
                elif 1.3 <= ar < 1.5:
                    tfm_imgs = two_crops(obj_img) + [pad_only(obj_img)]
                    n_2crops += 2
                    n_pad += 1
                elif 1.5 <= ar < 2.0:
                    tfm_imgs = two_crops(obj_img)
                    n_2crops += 2
                elif 1.0 < 1 / ar < 1.5:
                    tfm_imgs = [top_crop(obj_img), pad_only(obj_img)]
                    n_top += 1
                    n_pad += 1
                elif 1.5 <= 1 / ar < 1.8:
                    tfm_imgs = [top_crop(obj_img)]
                    n_top += 1
                else:
                    tfm_imgs = None
                    n_skip += 1
                add_sample_(tfm_imgs, label, 0)
                add_sample_(make_x2res(obj_img, bbox), label, 1)
                prev_bbox = bbox

                bbox = calc_bbox(o, img_w=img.width, img_h=img.height, zoom=medium_zoom, try_square=False)
                if bbox == prev_bbox:
                    n_dup += 1
                    continue
                if min(bb2wh(bbox)) < IMG_SIZE_2: continue
                obj_img = img.crop(bbox)
                ar = obj_img.width / obj_img.height
                if 1.3 < ar < 1.5:
                    tfm_imgs = two_crops(obj_img)
                    n_2crops += 2
                elif 1.05 < 1 / ar < 1.6:  # maybe tall
                    tfm_imgs = top_crop(obj_img)
                    n_top += 1
                else:
                    continue
                add_sample_(tfm_imgs, label, 0)
                add_sample_(make_x2res(obj_img, bbox), label, 1)
                prev_bbox = bbox

        n_x1, n_x2 = 0, 0
        for i, img in enumerate(imgs):
            if img.size == IMG_SIZE:
                n_x1 += 1
            else:
                n_x2 += 1

        print(f'Found {len(self.classes)} classes\nLoaded 64x64 {n_x1} images\n'
              f'Loaded 128x128 {n_x2} images\n')
        print(f'Pad only: {n_pad}\nCrop center: {n_center}\n'
              f'Crop top: {n_top}\nCrop 2 times: {n_2crops}\n'
              f'Take as-is: {n_noop}\nSkip: {n_skip}\nSame bbox: {n_dup}')
        return imgs, labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        tfms = self.transforms[img.tfm]
        img = tfms(image=img.px)['image']
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def create_runtime_tfms():
    mean, std = [0.5] * 3, [0.5] * 3
    resize_to_64 = A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_AREA)
    out = [A.HorizontalFlip(p=0.5), A.Normalize(mean=mean, std=std), ToTensor()]

    rand_crop = A.Compose([
        A.SmallestMaxSize(IMG_SIZE + 8, interpolation=cv2.INTER_AREA),
        A.RandomCrop(IMG_SIZE, IMG_SIZE)
    ])

    affine_1 = A.ShiftScaleRotate(
        shift_limit=0, scale_limit=0.1, rotate_limit=8,
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    affine_1 = A.Compose([affine_1, resize_to_64])

    affine_2 = A.ShiftScaleRotate(
        shift_limit=0.06, scale_limit=(-0.06, 0.18), rotate_limit=6,
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    affine_2 = A.Compose([affine_2, resize_to_64])

    tfm_0 = A.Compose(out)
    tfm_1 = A.Compose([A.OneOrOther(affine_1, rand_crop, p=1.0), *out])
    tfm_2 = A.Compose([affine_2, *out])
    return [tfm_0, tfm_1, tfm_2]


def get_data_loaders(data_root=None, label_root=None, batch_size=32, num_workers=2, shuffle=True,
                     pin_memory=True, drop_last=True):
    print('Using dataset root location %s' % data_root)
    train_set = DogsDataSet(data_root, label_root, create_runtime_tfms())
    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last}  # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders