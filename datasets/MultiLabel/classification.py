import os
import numpy as np
import random
import json

from datasets.bases import BaseImageDataset



class ZSMultiLabelClassification(BaseImageDataset):
    def __init__(self, root='', verbose=True, **kwargs):
        super(ZSMultiLabelClassification, self).__init__()
        self.dataset_dir = root
        if "coco" in root.lower():
            dataset_name = "COCO"
            self.train_file = os.path.join(self.dataset_dir, "annotations", 'train_48_filtered.json')
            self.test_file = os.path.join(self.dataset_dir, "annotations", 'test_17_filtered.json')
            self.test_file_gzsl = os.path.join(self.dataset_dir, "annotations", 'test_65_filtered.json')
        elif "nus" in root.lower():
            dataset_name = "NUS"
            self.train_file = os.path.join(self.dataset_dir, "annotations", 'train_925_filtered.json')
            self.test_file = os.path.join(self.dataset_dir, "annotations", 'test_81_filtered.json')
            self.test_file_gzsl = os.path.join(self.dataset_dir, "annotations", 'test_1006_filtered.json')
        else:
            raise NotImplementedError
        self._check_before_run()

        train, class2idx, name_train = self._load_dataset(self.dataset_dir, self.train_file, shuffle=True)
        test, _, name_test = self._load_dataset(self.dataset_dir, self.test_file, shuffle=False)
        test_gzsl, _, _ = self._load_dataset(self.dataset_dir, self.test_file_gzsl, shuffle=False, names=name_train+name_test)
        self.train = train
        self.test = test
        self.test_gzsl = test_gzsl
        self.class2idx = class2idx
        if verbose:
            print(f"=> {dataset_name} ZSL Dataset:")
            self.print_dataset_statistics(train, test)
            print(f"=> {dataset_name} GZSL Dataset:")
            self.print_dataset_statistics(train, test_gzsl)
        self.classnames_seen = name_train
        self.classnames_unseen = name_test
        self.classnames = name_train+name_test
        self.num_cls_train = len(name_train)
        self.num_cls_test = len(name_test)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not os.path.exists(self.test_file):
            raise RuntimeError("'{}' is not available".format(self.test_file))

    def _load_dataset(self, data_dir, annot_path, shuffle=True, names=None):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            classes = sorted(annotation['classes']) if names is None else names
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_info = annotation['images']
            img_wo_objects = 0
            for img_info in images_info:
                labels_idx = list()
                rel_image_path, img_labels = img_info
                full_image_path = os.path.join(data_dir, rel_image_path)
                labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
                labels_idx = list(set(labels_idx))
                # transform to one-hot
                onehot = np.zeros(len(classes), dtype=int)
                onehot[labels_idx] = 1
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, onehot))
        if img_wo_objects:
            print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        if shuffle:
            random.shuffle(out_data)
        return out_data, class_to_idx, classes



