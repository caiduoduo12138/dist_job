import os
import cv2
import time
import numpy as np
import torch


from PIL import Image
from torchvision.datasets import CocoDetection
import detection.transforms as T


current_file_path = os.path.abspath(__file__)
ROOT = current_file_path.split("test_io.py")[0]
REPEATS = 20
IMAGE_SIZE = (1333, 800)
BATCH_SIZE = 8


def read_image():
    io_time = 0.0
    t1 = time.time()
    file_dir = os.path.join(ROOT, "PennFudanPed/PNGImages")
    file_list = REPEATS * os.listdir(file_dir)
    for file_name in file_list:
        _ = cv2.imread(os.path.join(file_dir, file_name))
        # time.sleep(0.1)

    t2 = time.time()
    io_time += (t2 - t1)
    print("io_time: {} Seconds".format(round(io_time, 4)))
    return io_time


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.extend([T.RandomHorizontalFlip(0.5),
                           T.ScaleJitter(target_size=IMAGE_SIZE, scale_range=(0.8, 1.2))])
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


class PennFudanDataset(CocoDetection):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) * REPEATS
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks")))) * REPEATS

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_data_iteration_time():
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    t1 = time.time()
    for item in dataset:
        continue
    t2 = time.time()
    iteration_time = t2 - t1
    print("data_iteration_time: {} Seconds".format(round(iteration_time, 4)))
    return iteration_time


def get_dataloader_time():
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        collate_fn=collate_fn, shuffle=True, pin_memory=True)
    t1 = time.time()
    for item in data_loader:
        img, target = item
    t2 = time.time()
    dataloader_time = t2 - t1
    print("dataloader_time: {} Seconds".format(round(dataloader_time, 4)))


if __name__ == '__main__':
    read_image()
    get_data_iteration_time()
    get_dataloader_time()
