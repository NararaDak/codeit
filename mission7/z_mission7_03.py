#!/usr/bin/env python3
# train_pet_3class_bbox.py
# Oxford-IIIT Pet dataset (Kaggle) - Build classification dataset from XML bboxes
# Classes: 0=background, 1=cat, 2=dog
# Models: EfficientNet-B3 / Swin-Tiny / MobileNetV2 / MobileNetV3 (transfer learning)
# Usage example:
# python train_pet_3class_bbox.py --base_dir /path/to/pet_data --model EfficientNetB3 --epochs 8 --batch_size 32
#
# More Korean AI info: https://gptonline.ai/ko/

import os
import random
import time
import argparse
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from tqdm.auto import tqdm

# -------------------------
# Utilities
# -------------------------
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def makedirs(d):
    os.makedirs(d, exist_ok=True)

def iou_box(boxA, boxB):
    # boxes: [xmin,ymin,xmax,ymax]
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0

# -------------------------
# Read dataset meta
# -------------------------
def read_meta(base_dir):
    base_dir = str(base_dir)
    trainval_file = os.path.join(base_dir, "annotations", "annotations", "trainval.txt")
    test_file = os.path.join(base_dir, "annotations", "annotations", "test.txt")
    image_dir = os.path.join(base_dir, "images", "images")
    xml_dir = os.path.join(base_dir, "annotations", "annotations", "xmls")

    df_trainval = pd.read_csv(trainval_file, sep="\s+", header=None)
    df_trainval.columns = ["Image","ClassID","Species","BreedID"]
    df_test = pd.read_csv(test_file, sep="\s+", header=None)
    df_test.columns = ["Image","ClassID","Species","BreedID"]

    return df_trainval, df_test, image_dir, xml_dir

# -------------------------
# Build classification samples from XML bboxes
# -------------------------
def parse_xml_for_image(xml_path):
    """
    Return list of (class_name, bbox) for each object in the xml.
    bbox = [xmin, ymin, xmax, ymax] (int)
    class_name is the <name> field in XML (breed name)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        out.append((name, [xmin, ymin, xmax, ymax]))
    return out

def generate_classification_samples(df_trainval, df_test, image_dir, xml_dir, background_per_image=1, random_seed=42):
    """
    Build sample list for classification:
    - For each xml bbox: create a sample (image_path, bbox, label) where label {1=cat,2=dog}
      species mapping uses df_trainval/df_test 'Species' column: we assume Species value in those files corresponds to the image.
      We'll create a dict image->species from combined df.
    - For background: for each image attempt to sample `background_per_image` random crops that have IoU < 0.1 with ALL bboxes.
    Returns:
      samples: list of dicts: {'img': path, 'bbox': [xmin,ymin,xmax,ymax], 'label': 0/1/2}
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # build image -> species mapping from trainval+test
    df_all = pd.concat([df_trainval, df_test], ignore_index=True)
    img2species = dict(zip(df_all['Image'].astype(str), df_all['Species'].astype(int)))

    # gather xml files
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    samples = []

    for xml_file in tqdm(xml_files, desc="Parsing XMLs"):
        base = os.path.splitext(xml_file)[0]  # base name
        xml_path = os.path.join(xml_dir, xml_file)
        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            continue

        # parse objects
        objs = parse_xml_for_image(xml_path)
        bboxes = []
        for name, bbox in objs:
            bboxes.append(bbox)

            # determine species label: use image-level species mapping if available, otherwise fallback by
            # trying to detect 'cat' in breed name
            sp = img2species.get(base, None)
            if sp is None:
                # fallback: heuristic from breed name
                lab = 1 if 'cat' in name.lower() else 2  # cat=1, dog=2
            else:
                # NOTE: In Oxford-IIIT Pet dataset, Species: 1 = cat, 2 = dog
                lab = int(sp)  # 1 or 2
            samples.append({'img': img_path, 'bbox': bbox, 'label': lab})

        # background sampling: sample background_per_image crops that do not overlap significantly with any bbox
        # We'll sample crops of size similar to mean bbox (or fixed)
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        # determine target crop size: if there are bboxes use avg size else use 0.25 * min(W,H)
        if bboxes:
            widths = [bb[2]-bb[0] for bb in bboxes]
            heights = [bb[3]-bb[1] for bb in bboxes]
            cw = int(np.mean(widths))
            ch = int(np.mean(heights))
            # clamp
            cw = max(32, min(cw, W//2))
            ch = max(32, min(ch, H//2))
        else:
            cw = max(32, min(W//4, 256))
            ch = max(32, min(H//4, 256))

        attempts = 0
        created = 0
        max_attempts = 50 * background_per_image
        while created < background_per_image and attempts < max_attempts:
            attempts += 1
            # random top-left
            x1 = random.randint(0, max(0, W - cw))
            y1 = random.randint(0, max(0, H - ch))
            x2 = x1 + cw
            y2 = y1 + ch
            crop_box = [x1, y1, x2, y2]
            # compute IoU with all bboxes; if all < 0.1 accept
            ok = True
            for bb in bboxes:
                if iou_box(crop_box, bb) >= 0.1:
                    ok = False
                    break
            if ok:
                samples.append({'img': img_path, 'bbox': crop_box, 'label': 0})
                created += 1

    # shuffle
    random.shuffle(samples)
    return samples

# -------------------------
# Dataset for classification from samples
# -------------------------
class BBoxClassificationDataset(Dataset):
    """
    dataset built from samples list: each sample {'img':path,'bbox':[xmin,ymin,xmax,ymax],'label':0/1/2}
    On-the-fly crop and transform.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img']).convert("RGB")
        xmin, ymin, xmax, ymax = s['bbox']
        # safe clamp
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img.width, xmax); ymax = min(img.height, ymax)
        if xmax <= xmin or ymax <= ymin:
            # fallback to full image
            crop = img
        else:
            crop = img.crop((xmin, ymin, xmax, ymax))
        label = int(s['label'])
        if self.transform:
            crop = self.transform(crop)
        return crop, label

# -------------------------
# Transforms
# -------------------------
def get_transforms_for_model(model_name, is_train=True, target_size=224):
    # Model-specific recommended sizes: EfficientNetB3 -> 300, Swin/MobileNets -> 224
    if model_name == "EfficientNetB3":
        out_size = 300
    else:
        out_size = target_size

    if is_train:
        return T.Compose([
            T.RandomResizedCrop(out_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.15,0.15,0.15),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(out_size * 1.14)),
            T.CenterCrop(out_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

# -------------------------
# Model factory (transfer classes)
# -------------------------
class BasicTransfer(nn.Module):
    def __init__(self):
        super().__init__()

    def setup_training(self):
        raise NotImplementedError

class EfficientNetB3Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        # torchvision EfficientNet-B3
        try:
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.model = torchvision.models.efficientnet_b3(weights=weights)
        except Exception:
            self.model = torchvision.models.efficientnet_b3(pretrained=True)
        # replace classifier
        try:
            in_f = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.model.classifier.in_features, num_classes))
        self.gubun = gubun
        self.lr = lr
        self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, x): return self.model(x)

    def setup_training(self):
        head_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                p.requires_grad = True
                head_params.append(p)
            else:
                if self.gubun == "freeze":
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    backbone_params.append(p)
        groups = [{"params": head_params, "lr": self.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_ratio})
        opt = torch.optim.Adam(groups, lr=self.lr)
        return opt

class SwinTinyTransfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_t(weights=weights)
        except Exception:
            self.model = torchvision.models.swin_t(pretrained=True)
        in_f = self.model.head.in_features
        self.model.head = nn.Linear(in_f, num_classes)
        self.gubun = gubun
        self.lr = lr
        self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, x): return self.model(x)

    def setup_training(self):
        head_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if "head" in name:
                p.requires_grad = True
                head_params.append(p)
            else:
                if self.gubun == "freeze":
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    backbone_params.append(p)
        groups = [{"params": head_params, "lr": self.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

class MobileNetV2Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v2(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_f, num_classes)
        self.gubun = gubun; self.lr = lr; self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, x): return self.model(x)

    def setup_training(self):
        head_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True
                head_params.append(p)
            else:
                if self.gubun == "freeze":
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    backbone_params.append(p)
        groups = [{"params": head_params, "lr": self.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

class MobileNetV3Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v3_large(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # mobilenet_v3_large classifier usually classifier[3]
        try:
            in_f = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_f, num_classes)
        except Exception:
            # fallback: replace last layer
            if hasattr(self.model, "classifier"):
                last = list(self.model.classifier.children())[-1]
                in_f = last.in_features
                self.model.classifier[-1] = nn.Linear(in_f, num_classes)
        self.gubun = gubun; self.lr = lr; self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, x): return self.model(x)

    def setup_training(self):
        head_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True
                head_params.append(p)
            else:
                if self.gubun == "freeze":
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    backbone_params.append(p)
        groups = [{"params": head_params, "lr": self.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

def make_model(model_name, num_classes, gubun, lr, ratio):
    model_name = model_name.lower()
    if model_name == "efficientnetb3":
        model = EfficientNetB3Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
    elif model_name == "swintiny":
        model = SwinTinyTransfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
    elif model_name == "mobilenetv2":
        model = MobileNetV2Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
    elif model_name == "mobilenetv3":
        model = MobileNetV3Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
    else:
        raise ValueError("Unknown model_name: " + str(model_name))
    opt = model.setup_training()
    return model, opt

# -------------------------
# Training / Validation loops
# -------------------------
def train_one_epoch(model, opt, loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device); labels = labels.to(device)
        opt.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward(); opt.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    return running_loss / total, correct / total

def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            outs = model(imgs)
            loss = criterion(outs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total if total>0 else 0.0, correct / total if total>0 else 0.0

# -------------------------
# Main
# -------------------------
def main(args):
    print("Start:", now_str())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df_trainval, df_test, image_dir, xml_dir = read_meta(args.base_dir)

    # Build classification samples (bbox crops + background)
    print("-> Generating classification samples from XML bboxes (this may take a while)...")
    samples = generate_classification_samples(df_trainval, df_test, image_dir, xml_dir,
                                              background_per_image=args.bg_per_image,
                                              random_seed=args.seed)
    print(f"Total samples generated: {len(samples)}  (example: {samples[:3]})")

    # Split into train/val/test sets (stratified by label)
    df_samples = pd.DataFrame(samples)
    # ensure label column exists
    if 'label' not in df_samples.columns:
        print("No samples found. Exiting.")
        return

    # Train/Val split from df_samples based on labels
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df_samples, test_size=args.val_ratio, stratify=df_samples['label'], random_state=args.seed)
    # For test, we will use generated samples from df_test images only for fairness:
    test_samples = [s for s in samples if os.path.splitext(os.path.basename(s['img']))[0] in set(df_test['Image'].astype(str).tolist())]
    if len(test_samples) < 1:
        # fallback: use val as test if no explicit test samples
        test_df = val_df
    else:
        test_df = pd.DataFrame(test_samples)

    # Datasets + transforms
    train_transform = get_transforms_for_model(args.model, is_train=True, target_size=args.size)
    val_transform = get_transforms_for_model(args.model, is_train=False, target_size=args.size)
    train_ds = BBoxClassificationDataset(train_df.to_dict('records'), transform=train_transform)
    val_ds = BBoxClassificationDataset(val_df.to_dict('records'), transform=val_transform)
    test_ds = BBoxClassificationDataset(test_df.to_dict('records'), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Dataset sizes: train", len(train_ds), "val", len(val_ds), "test", len(test_ds))
    num_classes = 3  # background/cat/dog

    # Model
    model, opt = make_model(args.model, num_classes=num_classes, gubun=args.gubun, lr=args.lr, ratio=args.ratio)
    model = model.to(device)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    best_val_acc = 0.0
    out_dir = os.path.join(args.base_dir, "model_outs", f"{args.model}_{int(time.time())}")
    makedirs(out_dir)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, opt, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        took = time.time() - t0
        print(f"[{now_str()}] Epoch {epoch}/{args.epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} time {took:.1f}s")
        # save best
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": vars(args)
        }
        torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch_{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(out_dir, f"best.pth"))
            print("  >> Best model updated.")

    # Final test
    print("== Final test evaluation ==")
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    print("Outputs saved to:", out_dir)
    print("Done:", now_str())

# -------------------------
# Argument parser
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pet 3-class classification using bbox crops (background/cat/dog)")
    parser.add_argument("--base_dir", type=str, required=True, help="path to pet_data root (images/, annotations/...)")
    parser.add_argument("--model", type=str, default="EfficientNetB3", choices=["EfficientNetB3","SwinTiny","MobileNetV2","MobileNetV3"], help="model type")
    parser.add_argument("--gubun", type=str, default="freeze", choices=["freeze","partial","full"], help="freeze: only head, partial: head+backbone low lr, full: all trainable")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ratio", type=float, default=0.1, help="backbone lr ratio for partial training")
    parser.add_argument("--bg_per_image", type=int, default=1, help="number of background crops to sample per image")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="validation split ratio from generated samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=224, help="target input size for models (224 default; EfficientNetB3 uses 300 automatically)")
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    args = parser.parse_args()
    main(args)

# 끝 — 더 많은 한국어 AI 자료는 https://gptonline.ai/ko/ 에서 확인하세요 😊
