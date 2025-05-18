import os
import json
import numpy as np
import cv2
from xml.etree import ElementTree as et
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# --------------------------- CONFIG ---------------------------
class Config:
    fish = 'karp'
    img_size = (640, 640)
    batch_size = 3
    num_workers = 0
    num_epochs = 10
    train_dir = f'data/{fish}/train'
    test_dir = f'data/{fish}/test'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------- DATASET ---------------------------
class CellImagesDataset(Dataset):
    def __init__(self, files_dir, width, height, classes, transforms=None):
        self.files_dir = files_dir
        self.width = width
        self.height = height
        self.transforms = transforms
        self.classes = classes
        self.imgs = [img for img in sorted(os.listdir(files_dir)) if img.endswith('.jpg')]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        annot_path = os.path.join(self.files_dir, img_name.replace('.jpg', '.xml'))
        tree = et.parse(annot_path)
        root = tree.getroot()
        ht, wt = img.shape[:2]

        boxes, labels = [], []
        for obj in root.findall('object'):
            labels.append(self.classes.index(obj.find('name').text))
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) * self.width / wt
            xmax = int(bndbox.find('xmax').text) * self.width / wt
            ymin = int(bndbox.find('ymin').text) * self.height / ht
            ymax = int(bndbox.find('ymax').text) * self.height / ht
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd, "image_id": torch.tensor([idx])}

        if self.transforms:
            sample = self.transforms(image=img_res, bboxes=target['boxes'].tolist(), labels=labels.tolist())
            img_res = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

        return img_res, target

    def __len__(self):
        return len(self.imgs)


# --------------------------- TRANSFORMS ---------------------------
class Transforms:
    @staticmethod
    def get(train=True):
        if train:
            return A.Compose([
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.MotionBlur(p=0.2),
                A.HueSaturationValue(p=0.3),
                A.Resize(height=640, width=640),
                ToTensorV2(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            return A.Compose([
                A.Resize(height=640, width=640),
                ToTensorV2(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# --------------------------- MODEL FACTORY ---------------------------
class ModelFactory:
    @staticmethod
    def create(num_classes):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


# --------------------------- DATASET MANAGER ---------------------------
class DatasetManager:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    def get_classes(self):
        classes = set()
        for path in [self.train_dir, self.test_dir]:
            for file in os.listdir(path):
                if file.endswith('.xml'):
                    tree = et.parse(os.path.join(path, file))
                    for obj in tree.getroot().findall('object'):
                        classes.add(obj.find('name').text)
        return sorted(classes)

    def split_dataset(self, dataset, val_split=0.1):
        indices = np.random.permutation(len(dataset)).tolist()
        split_idx = int(len(dataset) * (1 - val_split))
        return Subset(dataset, indices[:split_idx]), Subset(dataset, indices[split_idx:])


# --------------------------- CLASS STATS ---------------------------
class ClassStats:
    def __init__(self, dataset, num_classes, device):
        self.dataset = dataset
        self.num_classes = num_classes
        self.device = device

    def compute_class_weights(self):
        counter = Counter()
        for _, target in self.dataset:
            counter.update(target["labels"].tolist())
        total = sum(counter.values())
        weights = torch.zeros(self.num_classes, device=self.device)
        for cls, freq in counter.items():
            weights[cls] = 1 + np.log(total / freq)
        return weights


# --------------------------- TRAINER ---------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, class_weights, device, log_dir, num_epochs=10):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.device = device
        self.num_epochs = num_epochs
        self.metric = MeanAveragePrecision()
        self.log_dir = log_dir

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self._train_epoch(epoch)
            self._validate(epoch)
        return self.model

    def _train_epoch(self, epoch):
        self.model.train()
        loss_accum = 0.0
        for images, targets in self.train_loader:
            if any(len(t["labels"]) == 0 for t in targets):
                continue
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            batch_weights = [self.class_weights[label].mean() for target in targets for label in target["labels"]]
            avg_weight = torch.tensor(batch_weights).mean().to(self.device)
            loss_weights = {'loss_classifier': avg_weight, 'loss_box_reg': 1.0, 'loss_objectness': 0.5, 'loss_rpn_box_reg': 0.5}
            loss = sum(loss_dict[k] * loss_weights.get(k, 1.0) for k in loss_dict)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_accum += loss.item()

        train_loss = loss_accum / len(self.train_loader)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        self.scheduler.step(train_loss)
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.log_dir}/pytorch_model-e{epoch}.pt")

    def _validate(self, epoch):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, targets_batch in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets_batch = [{k: v.to(self.device) for k, v in t.items()} for t in targets_batch]
                outputs = self.model(images)
                preds.extend(outputs)
                targets.extend(targets_batch)

        self.metric.update(preds, targets)
        val_map = self.metric.compute()
        print(f"[Epoch {epoch}] Val mAP: {val_map['map']}")


# --------------------------- MAIN ---------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    cfg = Config()
    dm = DatasetManager(cfg.train_dir, cfg.test_dir)
    classes = dm.get_classes()

    with open(f'data/{cfg.fish}/classes_idx.json', 'w') as f:
        json.dump({cls: i for i, cls in enumerate(classes)}, f)

    dataset = CellImagesDataset(cfg.train_dir, *cfg.img_size, classes, Transforms.get(train=True))
    dataset_test = CellImagesDataset(cfg.test_dir, *cfg.img_size, classes, Transforms.get(train=False))

    train_data, val_data = dm.split_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = ModelFactory.create(len(classes))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    class_weights = ClassStats(dataset, len(classes), cfg.device).compute_class_weights()

    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, class_weights, cfg.device, f"logs/{cfg.fish}", cfg.num_epochs)
    trainer.train()


if __name__ == "__main__":
    main()
