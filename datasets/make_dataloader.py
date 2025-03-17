import os
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

from .bases import ImageDataset
from .MultiLabel.classification import ZSMultiLabelClassification



def train_collate_fn(batch):
    imgs, label = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64) 
    return torch.stack(imgs, dim=0), label


def val_collate_fn(batch):
    imgs, label = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(imgs, dim=0), label


def make_dataloader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = ZSMultiLabelClassification(root=cfg.DATASETS.ROOT_DIR)

    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    train_set = ImageDataset(dataset.train, transform=train_transforms, mirror=True, Aug=True)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.test, transform=val_transforms, mirror=False, Aug=False)
    val_set_gzsl = ImageDataset(dataset.test_gzsl, transform=val_transforms, mirror=False, Aug=False)

    if cfg.MODEL.DIST_TRAIN:
        print('DIST_TRAIN START')
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        print('===========================\n mini batch size:', mini_batch_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        nw = min([os.cpu_count(), mini_batch_size if mini_batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=mini_batch_size,
            pin_memory=True,
            num_workers=nw,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=train_collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        val_loader_gzsl = DataLoader(
            val_set_gzsl, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        return train_loader, val_loader, val_loader_gzsl, train_sampler, dataset
    else:
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.SOLVER.IMS_PER_BATCH, 
            shuffle=False,
            num_workers=num_workers, 
            collate_fn=train_collate_fn, 
            drop_last=True, 
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )
        val_loader_gzsl = DataLoader(
            val_set_gzsl, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )
        return train_loader, val_loader, val_loader_gzsl, None, dataset

