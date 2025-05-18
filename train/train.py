import os
import json
import yaml
import numpy as np
import cv2
from xml.etree import ElementTree as et
from collections import Counter

from loguru import logger
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



# --------------------------- CONFIG ---------------------------
class Config:
    def __init__(self, yaml_path="train/config.yaml"):
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        self.fish = yaml_config['fish']
        self.img_size = tuple(yaml_config['img_size'])
        self.batch_size = yaml_config['batch_size']
        self.num_workers = min(yaml_config["num_workers"], os.cpu_count() or 1)
        self.num_epochs = yaml_config['num_epochs']
        self.lr = yaml_config['lr']
        self.weight_decay = yaml_config['weight_decay']
        self.momentum = yaml_config['momentum']
        self.train_dir = yaml_config['paths']['train_dir']
        self.test_dir = yaml_config['paths']['test_dir']
        self.log_dir = yaml_config['paths']['log_dir']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------- UTILITIES ---------------------------
def parse_classes_from_annotations(paths: list[str]) -> list[str]:
    classes = set()
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.xml'):
                tree = et.parse(os.path.join(path, file))
                for obj in tree.getroot().findall('object'):
                    classes.add(obj.find('name').text)
    return sorted(classes)

def save_class_index_map(fish: str, classes: list[str]):
    with open(f'data/{fish}/classes_idx.json', 'w') as f:
        json.dump({cls: i for i, cls in enumerate(classes)}, f)

def collate_fn(batch):
    return tuple(zip(*batch))


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
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA) / 255.0

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
    def get(train=True, img_size=(640, 640)):
        base = [A.Resize(*img_size), ToTensorV2()]
        if not train:
            return A.Compose(base, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        aug = [
            A.RandomBrightnessContrast(p=0.3),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.MotionBlur(p=0.2),
            A.HueSaturationValue(p=0.3)
        ]
        return A.Compose(aug + base, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# --------------------------- MODEL FACTORY ---------------------------
class ModelFactory:
    @staticmethod
    def create(num_classes):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


# --------------------------- TRAINER ---------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, class_weights, device, log_dir, num_epochs):
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

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # log_file = os.path.join(self.log_dir, f"training_{timestamp}.log")
        #
        # logger.add(
        #     log_file,
        #     format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} - {message}",
        #     level="INFO"
        # )
        #
        # logger.info(f"Logging to file: {log_file}")

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"[Epoch {epoch}]")
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.scheduler.step()
            torch.save(self.model.state_dict(), f"{self.log_dir}/pytorch_model-e{epoch}.pt")
        return self.model

    def calculate_metrics(self, preds, targets, prefix=""):
        self.metric.reset()
        self.metric.update(preds, targets)
        results = self.metric.compute()
        map_value = results['map'].item()
        map_50 = results['map_50'].item()
        map_75 = results['map_75'].item()
        logger.info(f"{prefix}mAP: {map_value:.4f}, mAP@50: {map_50:.4f}, mAP@75: {map_75:.4f}")
        return {'map': map_value, 'map_50': map_50, 'map_75': map_75}

    def process_epoch(self, data_loader, epoch, is_training=False):
        self.current_epoch = epoch
        total_loss = 0.0
        total_classifier_loss = 0.0
        total_objectness_loss = 0.0
        preds, targets_all = [], []
        valid_batches = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for images, targets in data_loader:
                if any(len(t["labels"]) == 0 for t in targets):
                    continue
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                was_training = self.model.training
                self.model.train()
                loss_dict = self.model(images, targets)
                if not is_training:
                    self.model.eval()
                elif not was_training:
                    self.model.train()

                batch_weights = [self.class_weights[label].mean() for target in targets for label in target["labels"]]
                avg_weight = torch.tensor(batch_weights).mean().to(self.device) if batch_weights else 1.0
                loss_weights = {
                    'loss_classifier': avg_weight,
                    'loss_box_reg': 1.0,
                    'loss_objectness': 0.5,
                    'loss_rpn_box_reg': 0.5
                }
                loss = sum(loss_dict[k] * loss_weights.get(k, 1.0) for k in loss_dict)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_classifier_loss += loss_dict.get('loss_classifier', torch.tensor(0.0, device=self.device)).item()
                total_objectness_loss += loss_dict.get('loss_objectness', torch.tensor(0.0, device=self.device)).item()
                valid_batches += 1

                with torch.no_grad():
                    if self.model.training:
                        self.model.eval()
                    outputs = self.model(images)
                    if is_training:
                        self.model.train()
                    preds.extend(outputs)
                    targets_all.extend(targets)

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        avg_classifier_loss = total_classifier_loss / valid_batches if valid_batches > 0 else 0.0
        avg_objectness_loss = total_objectness_loss / valid_batches if valid_batches > 0 else 0.0
        return avg_loss, avg_classifier_loss, avg_objectness_loss, preds, targets_all

    def _train_epoch(self, epoch):
        avg_loss, avg_classifier_loss, avg_objectness_loss, preds, targets_all = self.process_epoch(
            self.train_loader, epoch, is_training=True
        )

        self.calculate_metrics(preds, targets_all, prefix="Train ")
        logger.info(f"Train Classifier Loss: {avg_classifier_loss:.4f}, Train Objectness Loss: {avg_objectness_loss:.4f}")
        return avg_loss

    def _validate_epoch(self, epoch):
        avg_loss, avg_classifier_loss, avg_objectness_loss, preds, targets_all = self.process_epoch(
            self.val_loader, epoch, is_training=False
        )

        self.calculate_metrics(preds, targets_all, prefix="Val ")
        logger.info(f"Val Classifier Loss: {avg_classifier_loss:.4f}, Val Objectness Loss: {avg_objectness_loss:.4f}")
        return avg_loss


# --------------------------- STATS ---------------------------
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


# --------------------------- MAIN ---------------------------
def get_dataloader(data, cfg, shuffle):
    return DataLoader(data, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, collate_fn=collate_fn)

def run_training(cfg: Config):
    classes = parse_classes_from_annotations([cfg.train_dir, cfg.test_dir])
    save_class_index_map(cfg.fish, classes)

    train_ds = CellImagesDataset(cfg.train_dir, *cfg.img_size, classes, Transforms.get(True, cfg.img_size))
    # test_ds = CellImagesDataset(cfg.test_dir, *cfg.img_size, classes, Transforms.get(False, cfg.img_size))

    np.random.seed(42)
    indices = np.random.permutation(len(train_ds)).tolist()
    split_idx = int(len(train_ds) * 0.9)
    train_data = Subset(train_ds, indices[:split_idx])
    val_data = Subset(train_ds, indices[split_idx:])

    train_loader = get_dataloader(train_data, cfg, shuffle=True)
    val_loader = get_dataloader(val_data, cfg, shuffle=False)

    model = ModelFactory.create(len(classes))

    params = [
        {"params": model.backbone.parameters(), "lr": cfg.lr / 10},
        {"params": model.rpn.parameters(), "lr": cfg.lr},
        {"params": model.roi_heads.parameters(), "lr": cfg.lr},
    ]

    optimizer = torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    class_weights = ClassStats(train_ds, len(classes), cfg.device).compute_class_weights()

    trainer = Trainer(
        model, optimizer, scheduler, train_loader, val_loader, class_weights, cfg.device, cfg.log_dir, cfg.num_epochs
    )
    trainer.train()

if __name__ == "__main__":
    config = Config()
    run_training(config)