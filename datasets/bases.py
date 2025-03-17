import os
import cv2
import numpy as np

from PIL import Image, ImageFile
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True



class BaseDataset(object):
    """
    Base class of reid dataset
    """
    def get_imagedata_info(self, data):
        imgs = []
        labels = []
        for data_img, data_label in data:
            imgs += [data_img]
            labels += [data_label]
        num_imgs = len(imgs)
        num_labels = len(labels)
        return num_imgs, num_labels

    def print_dataset_statistics(self):
        raise NotImplementedError



class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """
    def print_dataset_statistics(self, train, test):
        num_train_imgs, num_train_labels = self.get_imagedata_info(train)
        num_test_imgs, num_test_labels = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # images | # labels")
        print("  ----------------------------------------")
        print("  train    | {:8d} | {:9d}".format(num_train_imgs, num_train_labels))
        print("  test    | {:8d} | {:9d}".format(num_test_imgs, num_test_labels))
        print("  ----------------------------------------")



class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, shuffle = True, mirror=False, Aug = False):
        self.dataset = dataset
        self.transform = transform
        self.mirror = mirror
        self.Aug = Aug
        self.AugSeq = iaa.Sequential([
                iaa.Sometimes(0.2,
                    iaa.Crop(percent=(0, 0.1)),
                ),
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.2,
                    iaa.GaussianBlur(sigma=(0, 0.2))
                ),

                # Strengthen or weaken the contrast in each image.
                iaa.Sometimes(0.2, iaa.ContrastNormalization((0.75, 1.25))),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.015*255), per_channel=0.2),
                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Sometimes(0.2,
                              iaa.Affine(
                                  scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                  translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                  rotate=(-10, 10)
                              )
                ),
            ], random_order=True) # apply augmenters in random order

    def __getitem__(self, item):
        img_path, atts_list = self.dataset[item]
        img = Image.open(img_path).convert("RGB")

        # flip
        flip = 0
        if self.mirror:
            flip = np.random.choice(2)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.Aug:
            cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv_img = cv_img.reshape(1, cv_img.shape[0], cv_img.shape[1], cv_img.shape[2])
            cv_img = self.AugSeq.augment_images(cv_img)
            cv_img = cv_img.reshape(cv_img.shape[1], cv_img.shape[2], cv_img.shape[3])
            img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)
        return img, atts_list
    
    def __len__(self):
        return len(self.dataset)

