# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets import CocoDetection
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from detection.engine import train_one_epoch, evaluate
from detection.backbone import maskrcnn_resnet_x101_32_8d_fpn
import detection.utils as utils
import detection.transforms as T

#import torch.backends.cudnn as cudnn
#cudnn.deterministic = False
#cudnn.benchmark = True
# torch.backends.cudnn.enabled = True


IMAGE_SIZE = (1333, 800)
BATCH_SIZE = 2  # data per GPU
EPOCHS = 1
REPEATS = 20  # data length = 120
PRINT_FREQ = 10


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
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        mask = Image.open(mask_path)
        # mask = cv2.imread(mask_path)
        # mask = cv2.resize(mask, (800, 800), interpolation=cv2.INTER_AREA)
        # mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
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


def get_model_instance_segmentation(num_classes):
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model = maskrcnn_resnet_x101_32_8d_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transform(train):
    transforms = [T.ToTensor()]
    # if train:
    #     transforms.extend([T.RandomHorizontalFlip(0.5),
    #                        T.ScaleJitter(target_size=IMAGE_SIZE, scale_range=(1.0, 1.0))])
    return T.Compose(transforms)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # 这里只使用了数据中的一部分进行测试
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=True,
    )

    # 定义训练和测试数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        collate_fn=utils.collate_fn, sampler=train_sampler, pin_memory=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #     collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # 将模型放到响应的设备上
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 创建一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 设置学习率和调整策略
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = EPOCHS
    cuda_time = 0.0
    start_time = time.time()
    dataloader_time = 0.0
    for epoch in range(num_epochs):
        _, data_time, gpu_time = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=PRINT_FREQ)
        # 调整学习率
        lr_scheduler.step()
        # 评估模型
        # evaluate(model, data_loader_test, device=device)
        cuda_time += gpu_time
        dataloader_time += data_time

    end_time = time.time()
    total_time = end_time - start_time
    cpu_time = total_time - cuda_time

    if int(os.environ["RANK"]) == 0:
        print("--------------result--------------")
        print("    dataloader_time: {} Seconds".format(round(dataloader_time, 4 )))
        print("    cpu_time: {} Seconds".format(round(cpu_time, 4)))
        print("    cuda_time: {} Seconds".format(round(cuda_time, 4)))
        print("    total_time: {} Seconds".format(round(total_time, 4)))
        print("--------------finsh--------------")

    dist.barrier()  # 确保所有进程都停止于此
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
