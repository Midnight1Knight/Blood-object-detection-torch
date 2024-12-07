import os
import numpy as np

import cv2
from xml.etree import ElementTree as et

import torch
import torchvision

from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms

import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CellImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width

        self.imgs = [image for image in sorted(os.listdir(files_dir))
                     if image[-4:] == '.jpg']

        self.classes = classes

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)

        img_res /= 255.0

        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        wt = img.shape[1]
        ht = img.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_corr = (xmin / wt) * self.width
            xmax_corr = (xmax / wt) * self.width
            ymin_corr = (ymin / ht) * self.height
            ymax_corr = (ymax / ht) * self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # if self.transforms:
        #     sample = self.transforms(image=img_res,
        #                              bboxes=target['boxes'],
        #                              labels=labels)
        #
        #     img_res = sample['image']
        #     target['boxes'] = torch.Tensor(sample['bboxes'])
        if self.transforms:
            sample = self.transforms(image=img_res, bboxes=target['boxes'].tolist(), labels=labels.tolist())
            img_res = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def get_object_detection_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# def get_transform(train):
#     if train:
#         return A.Compose([
#             A.HorizontalFlip(0.5),
#             A.VerticalFlip(0.3),
#             ToTensorV2(p=1.0)
#         ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
#     else:
#         return A.Compose([
#             ToTensorV2(p=1.0)
#         ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.3),
            A.Resize(224, 224),  # Ensure resizing is explicit
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(224, 224),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_classes(path):
    classes = set()
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            with open(os.path.join(path, filename)) as f:
                tree = et.parse(f)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    classes.add(name)
    return list(classes)


train_dir = 'data/karp/train'
test_dir = 'data/karp/test'

train_classes = get_classes(train_dir)
test_classes = get_classes(test_dir)
classes = list(set(train_classes + test_classes))


dataset = CellImagesDataset(train_dir, 224, 224, classes=classes, transforms=get_transform(train=True))
dataset_test = CellImagesDataset(test_dir, 224, 224, classes=classes, transforms=get_transform(train=False))

torch.manual_seed(1)
np.random.seed(1)
indices = np.random.permutation(len(dataset)).tolist()

val_split = 0.1
tsize = len(dataset) - int(len(dataset) * val_split)

train_data = torch.utils.data.Subset(dataset, indices[:tsize])
val_data = torch.utils.data.Subset(dataset, indices[tsize:])


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=5, shuffle=True, num_workers=2,
    collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=5, shuffle=True, num_workers=2,
    collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=5, shuffle=False, num_workers=2,
    collate_fn=collate_fn)

n_batches, n_batches_test = len(train_loader), len(test_loader)
images, targets = next(iter(train_loader))

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

num_classes = len(train_classes)

model = get_object_detection_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


metric = MeanAveragePrecision()

# Define the lists to store the loss values for training and validation
train_loss_values = []
val_loss_values = []


def train_model(model, data_loader=None, num_epoch=10):
    for epoch in range(1, num_epoch + 1):
        print(f"Starting epoch {epoch} of {num_epoch}")

        time_start = time.time()
        loss_accum = 0.0
        # loss_mask_accum = 0.0
        loss_classifier_accum = 0.0

        model.train()

        for batch_idx, (images, targets) in enumerate(data_loader, 1):

            # Predict
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            # loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            # loss_mask_accum += loss_mask
            loss_classifier_accum += loss_dict['loss_classifier'].item()

            if batch_idx % 500 == 0:
                print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}.")

        lr_scheduler.step()

        # Train losses
        train_loss = loss_accum / n_batches

        # Store the loss value for training
        train_loss_values.append(train_loss)

        # train_loss_mask = loss_mask_accum / n_batches
        train_loss_classifier = loss_classifier_accum / n_batches

        elapsed = time.time() - time_start

        torch.save(model.state_dict(), f"logs/pytorch_model-e{epoch}.pt")
        prefix = f"[Epoch {epoch:2d} / {num_epoch:2d}]"
        # print(prefix)
        # print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
        print(f"{prefix} Train loss: {train_loss:7.3f} [{elapsed:.0f} secs]", end=' | ')

        preds_single = []
        targets_single = []

        for batch_idx, (images, targets) in enumerate(val_loader, 1):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            targets_single.extend(targets)

            model.eval()
            with torch.no_grad():
                pred = model(images)

            preds_single.extend(pred)

        metric.update(preds_single, targets_single)
        batch_map = metric.compute()
        val_loss_values.append(batch_map)
        print(f"Val mAP: {batch_map['map']}")

    return model


num_epoch = 10
model = train_model(model, train_loader, num_epoch)


metric_test = MeanAveragePrecision()

preds_single = []
targets_single = []

for batch_idx, (images, targets) in enumerate(test_loader, 1):

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    targets_single.extend(targets)

    model.eval()
    with torch.no_grad():
        pred = model(images)

    preds_single.extend(pred)

metric_test.update(preds_single, targets_single)
test_map = metric.compute()

print(f"Test mAP: {test_map['map']}")



