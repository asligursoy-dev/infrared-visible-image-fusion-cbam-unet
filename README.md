from google.colab import drive
drive.mount('/content/drive')

import os
DATASET_DIR = "/content/drive/MyDrive/LLVIP"
print(os.listdir(DATASET_DIR))

visible_train_dir = os.path.join(DATASET_DIR, "visible", "train")
visible_test_dir  = os.path.join(DATASET_DIR, "visible", "test")

infrared_train_dir = os.path.join(DATASET_DIR, "infrared", "train")
infrared_test_dir  = os.path.join(DATASET_DIR, "infrared", "test")

print("Visible train:", len(os.listdir(visible_train_dir)))
print("Visible test:", len(os.listdir(visible_test_dir)))
print("Infrared train:", len(os.listdir(infrared_train_dir)))
print("Infrared test:", len(os.listdir(infrared_test_dir)))

print("\nSample visible train files:")
print(os.listdir(visible_train_dir)[:5])

print("\nSample infrared train files:")
print(os.listdir(infrared_train_dir)[:5])

import os
import shutil
import random

# yeni dataset klasörü
YOLO_DIR = "/content/drive/MyDrive/LLVIP_YOLO"

train_img_dir = os.path.join(YOLO_DIR, "images", "train")
val_img_dir   = os.path.join(YOLO_DIR, "images", "val")

train_lbl_dir = os.path.join(YOLO_DIR, "labels", "train")
val_lbl_dir   = os.path.join(YOLO_DIR, "labels", "val")

# klasörleri oluştur
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# tüm image dosyaları
all_images = [f for f in os.listdir(visible_train_dir) if f.endswith(".jpg")]

# karıştır
random.shuffle(all_images)

# küçük subset kullanacağız
subset = all_images[:1000]

# train-val split
split_idx = int(0.8 * len(subset))

train_files = subset[:split_idx]
val_files   = subset[split_idx:]

labels_dir = os.path.join(DATASET_DIR, "labels_all")

def copy_files(file_list, img_target, lbl_target):
    for img_file in file_list:

        # image copy
        src_img = os.path.join(visible_train_dir, img_file)
        dst_img = os.path.join(img_target, img_file)

        shutil.copy(src_img, dst_img)

        # label copy
        label_file = img_file.replace(".jpg", ".txt")

        src_lbl = os.path.join(labels_dir, label_file)

        if os.path.exists(src_lbl):
            dst_lbl = os.path.join(lbl_target, label_file)
            shutil.copy(src_lbl, dst_lbl)

# copy işlemleri
copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print("Train images:", len(os.listdir(train_img_dir)))
print("Val images:", len(os.listdir(val_img_dir)))

import os, shutil, random

YOLO_DIR = "/content/drive/MyDrive/LLVIP_YOLO"

# Eski klasörü tamamen sil
if os.path.exists(YOLO_DIR):
    shutil.rmtree(YOLO_DIR)

train_img_dir = os.path.join(YOLO_DIR, "images", "train")
val_img_dir   = os.path.join(YOLO_DIR, "images", "val")
train_lbl_dir = os.path.join(YOLO_DIR, "labels", "train")
val_lbl_dir   = os.path.join(YOLO_DIR, "labels", "val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

labels_dir = os.path.join(DATASET_DIR, "labels_all")

all_images = [f for f in os.listdir(visible_train_dir) if f.endswith(".jpg")]
random.seed(42)
random.shuffle(all_images)

subset = all_images[:1000]
train_files = subset[:800]
val_files = subset[800:1000]

def copy_files(file_list, img_source_dir, img_target_dir, lbl_target_dir):
    copied = 0
    missing_labels = 0

    for img_file in file_list:
        label_file = img_file.replace(".jpg", ".txt")

        src_img = os.path.join(img_source_dir, img_file)
        src_lbl = os.path.join(labels_dir, label_file)

        if not os.path.exists(src_lbl):
            missing_labels += 1
            continue

        shutil.copy(src_img, os.path.join(img_target_dir, img_file))
        shutil.copy(src_lbl, os.path.join(lbl_target_dir, label_file))
        copied += 1

    return copied, missing_labels

train_copied, train_missing = copy_files(train_files, visible_train_dir, train_img_dir, train_lbl_dir)
val_copied, val_missing = copy_files(val_files, visible_train_dir, val_img_dir, val_lbl_dir)

print("Train copied:", train_copied, "Missing labels:", train_missing)
print("Val copied:", val_copied, "Missing labels:", val_missing)

print("Train images:", len(os.listdir(train_img_dir)))
print("Train labels:", len(os.listdir(train_lbl_dir)))
print("Val images:", len(os.listdir(val_img_dir)))
print("Val labels:", len(os.listdir(val_lbl_dir)))

yaml_path = "/content/drive/MyDrive/LLVIP_YOLO/llvip.yaml"

yaml_content = """
path: /content/drive/MyDrive/LLVIP_YOLO
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("YAML file created at:", yaml_path)

!pip install ultralytics -q

from ultralytics import YOLO
print("YOLOv8 imported successfully")

from ultralytics import YOLO

# küçük model
model = YOLO("yolov8n.pt")

# eğitim
model.train(
    data="/content/drive/MyDrive/LLVIP_YOLO/llvip.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0
)

import os
import shutil

IR_YOLO_DIR = "/content/drive/MyDrive/LLVIP_IR_YOLO"

# eski klasörü sil
if os.path.exists(IR_YOLO_DIR):
    shutil.rmtree(IR_YOLO_DIR)

# klasörleri oluştur
train_img_dir = os.path.join(IR_YOLO_DIR, "images", "train")
val_img_dir   = os.path.join(IR_YOLO_DIR, "images", "val")

train_lbl_dir = os.path.join(IR_YOLO_DIR, "labels", "train")
val_lbl_dir   = os.path.join(IR_YOLO_DIR, "labels", "val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# önceki splitleri kullan
def copy_ir_files(file_list, img_target, lbl_target):

    for img_file in file_list:

        src_img = os.path.join(infrared_train_dir, img_file)

        if not os.path.exists(src_img):
            continue

        shutil.copy(src_img, os.path.join(img_target, img_file))

        label_file = img_file.replace(".jpg", ".txt")

        src_lbl = os.path.join(labels_dir, label_file)

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, os.path.join(lbl_target, label_file))

copy_ir_files(train_files, train_img_dir, train_lbl_dir)
copy_ir_files(val_files, val_img_dir, val_lbl_dir)

print("IR train images:", len(os.listdir(train_img_dir)))
print("IR val images:", len(os.listdir(val_img_dir)))

yaml_path = "/content/drive/MyDrive/LLVIP_IR_YOLO/llvip_ir.yaml"

yaml_content = """
path: /content/drive/MyDrive/LLVIP_IR_YOLO
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("IR YAML created.")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/content/drive/MyDrive/LLVIP_IR_YOLO/llvip_ir.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0
)

visible_files = sorted(os.listdir(visible_train_dir))
infrared_files = sorted(os.listdir(infrared_train_dir))

print("Visible sample:")
print(visible_files[:10])

print("\nInfrared sample:")
print(infrared_files[:10])

import cv2
import numpy as np
import os

FUSED_DIR = "/content/drive/MyDrive/LLVIP_FUSED"

train_fused_dir = os.path.join(FUSED_DIR, "images", "train")
val_fused_dir   = os.path.join(FUSED_DIR, "images", "val")

os.makedirs(train_fused_dir, exist_ok=True)
os.makedirs(val_fused_dir, exist_ok=True)

def fuse_images(rgb_path, ir_path):

    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path)

    rgb = cv2.resize(rgb, (640, 512))
    ir = cv2.resize(ir, (640, 512))

    fused = cv2.addWeighted(rgb, 0.5, ir, 0.5, 0)

    return fused

    # train fusion images

for img_file in train_files:

    rgb_path = os.path.join(visible_train_dir, img_file)
    ir_path  = os.path.join(infrared_train_dir, img_file)

    if not os.path.exists(rgb_path):
        continue

    if not os.path.exists(ir_path):
        continue

    fused = fuse_images(rgb_path, ir_path)

    save_path = os.path.join(train_fused_dir, img_file)

    cv2.imwrite(save_path, fused)

print("Train fusion images created.")

for img_file in val_files:

    rgb_path = os.path.join(visible_train_dir, img_file)
    ir_path  = os.path.join(infrared_train_dir, img_file)

    if not os.path.exists(rgb_path):
        continue

    if not os.path.exists(ir_path):
        continue

    fused = fuse_images(rgb_path, ir_path)

    save_path = os.path.join(val_fused_dir, img_file)

    cv2.imwrite(save_path, fused)

print("Validation fusion images created.")

print("Train fused:", len(os.listdir(train_fused_dir)))
print("Val fused:", len(os.listdir(val_fused_dir)))

import shutil

train_fused_label_dir = os.path.join(FUSED_DIR, "labels", "train")
val_fused_label_dir   = os.path.join(FUSED_DIR, "labels", "val")

os.makedirs(train_fused_label_dir, exist_ok=True)
os.makedirs(val_fused_label_dir, exist_ok=True)

# train labels
for img_file in train_files:

    label_file = img_file.replace(".jpg", ".txt")

    src_label = os.path.join(labels_dir, label_file)

    if os.path.exists(src_label):

        shutil.copy(
            src_label,
            os.path.join(train_fused_label_dir, label_file)
        )

# val labels
for img_file in val_files:

    label_file = img_file.replace(".jpg", ".txt")

    src_label = os.path.join(labels_dir, label_file)

    if os.path.exists(src_label):

        shutil.copy(
            src_label,
            os.path.join(val_fused_label_dir, label_file)
        )

print("Fusion labels copied.")

yaml_path = "/content/drive/MyDrive/LLVIP_FUSED/fused.yaml"

yaml_content = """
path: /content/drive/MyDrive/LLVIP_FUSED
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("Fused YAML created.")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/content/drive/MyDrive/LLVIP_FUSED/fused.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0
)

import os
import cv2
import shutil

# Ana yollar
DATASET_DIR = "/content/drive/MyDrive/LLVIP"

visible_train_dir = os.path.join(DATASET_DIR, "visible", "train")
infrared_train_dir = os.path.join(DATASET_DIR, "infrared", "train")
labels_dir = os.path.join(DATASET_DIR, "labels_all")

# Daha önce kullandığımız train_files ve val_files listeleri varsa direkt çalışır.
# Eğer runtime sıfırlandıysa, aşağıdaki kısmı aç:
# import random
# all_images = sorted([f for f in os.listdir(visible_train_dir) if f.endswith(".jpg")])
# random.seed(42)
# random.shuffle(all_images)
# subset = all_images[:1000]
# train_files = subset[:800]
# val_files = subset[800:1000]

fusion_settings = [
    ("FUSED_05_05", 0.5, 0.5),
    ("FUSED_07_03", 0.7, 0.3),
    ("FUSED_03_07", 0.3, 0.7),
]

def create_weighted_fusion_dataset(folder_name, rgb_weight, ir_weight):
    output_dir = f"/content/drive/MyDrive/LLVIP_{folder_name}"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    def fuse_and_save(file_list, img_out_dir, lbl_out_dir):
        for img_file in file_list:
            rgb_path = os.path.join(visible_train_dir, img_file)
            ir_path = os.path.join(infrared_train_dir, img_file)
            label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))

            if not os.path.exists(rgb_path) or not os.path.exists(ir_path) or not os.path.exists(label_path):
                continue

            rgb = cv2.imread(rgb_path)
            ir = cv2.imread(ir_path)

            rgb = cv2.resize(rgb, (640, 512))
            ir = cv2.resize(ir, (640, 512))

            fused = cv2.addWeighted(rgb, rgb_weight, ir, ir_weight, 0)

            cv2.imwrite(os.path.join(img_out_dir, img_file), fused)
            shutil.copy(label_path, os.path.join(lbl_out_dir, img_file.replace(".jpg", ".txt")))

    fuse_and_save(train_files, train_img_dir, train_lbl_dir)
    fuse_and_save(val_files, val_img_dir, val_lbl_dir)

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

names:
  0: person
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Created: {output_dir}")
    print("Train images:", len(os.listdir(train_img_dir)))
    print("Val images:", len(os.listdir(val_img_dir)))
    print("YAML:", yaml_path)
    print("-" * 50)

for folder_name, rgb_w, ir_w in fusion_settings:
    create_weighted_fusion_dataset(folder_name, rgb_w, ir_w)

    from ultralytics import YOLO

datasets = [
    "/content/drive/MyDrive/LLVIP_FUSED_05_05/dataset.yaml",
    "/content/drive/MyDrive/LLVIP_FUSED_07_03/dataset.yaml",
    "/content/drive/MyDrive/LLVIP_FUSED_03_07/dataset.yaml",
]

for data_yaml in datasets:
    print("Training:", data_yaml)

    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        device=0
    )

    import pandas as pd

results = pd.DataFrame({
    "Method": [
        "Visible-only",
        "Average Fusion (0.5 RGB / 0.5 IR)",
        "Weighted Fusion (0.3 RGB / 0.7 IR)",
        "Weighted Fusion (0.7 RGB / 0.3 IR)"
    ],
    "Precision": [0.907, 0.938, 0.899, 0.935],
    "Recall": [0.734, 0.843, 0.831, 0.884],
    "mAP50": [0.812, 0.920, 0.894, 0.931],
    "mAP50-95": [0.401, 0.531, 0.498, 0.555]
})

results

import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_DIR = "/content/drive/MyDrive/LLVIP_Project_Results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(results["Method"], results["mAP50"])

plt.title("mAP50 Comparison Across Detection Inputs", fontsize=14, fontweight="bold")
plt.ylabel("mAP50", fontsize=12)
plt.xlabel("Method", fontsize=12)
plt.ylim(0, 1.0)
plt.xticks(rotation=25, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mAP50_comparison.png"), dpi=300)
plt.show()

import cv2
import matplotlib.pyplot as plt
import random

VISUAL_DIR = "/content/drive/MyDrive/LLVIP_Project_Results/visual_examples"
os.makedirs(VISUAL_DIR, exist_ok=True)

sample_files = val_files[:5]

for img_file in sample_files:
    rgb_path = os.path.join(visible_train_dir, img_file)
    ir_path = os.path.join(infrared_train_dir, img_file)
    fused_path = os.path.join("/content/drive/MyDrive/LLVIP_FUSED_07_03/images/val", img_file)

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    ir = cv2.cvtColor(cv2.imread(ir_path), cv2.COLOR_BGR2RGB)
    fused = cv2.cvtColor(cv2.imread(fused_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Visible Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ir)
    plt.title("Infrared Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fused)
    plt.title("Fused Image (0.7 RGB / 0.3 IR)")
    plt.axis("off")

    plt.tight_layout()
    save_name = img_file.replace(".jpg", "_fusion_example.png")
    plt.savefig(os.path.join(VISUAL_DIR, save_name), dpi=300)
    plt.show()

    from ultralytics import YOLO
import os

best_model_path = "/content/runs/detect/train-6/weights/best.pt"
model = YOLO(best_model_path)

source_dir = "/content/drive/MyDrive/LLVIP_FUSED_07_03/images/val"

pred_results = model.predict(
    source=source_dir,
    imgsz=640,
    conf=0.25,
    save=True,
    project="/content/drive/MyDrive/LLVIP_Project_Results",
    name="prediction_examples"
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class FusionDataset(Dataset):

    def __init__(self, visible_dir, infrared_dir, file_list):

        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.file_list = file_list

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img_file = self.file_list[idx]

        vis_path = os.path.join(self.visible_dir, img_file)
        ir_path  = os.path.join(self.infrared_dir, img_file)

        vis = cv2.imread(vis_path)
        ir  = cv2.imread(ir_path)

        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        ir  = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        vis = cv2.resize(vis, (256, 256))
        ir  = cv2.resize(ir, (256, 256))

        vis = self.transform(vis)
        ir  = self.transform(ir)

        return vis, ir

        class CBAM(nn.Module):

    def __init__(self, channels):

        super(CBAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):

        # channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        sa_input = torch.cat([avg_pool, max_pool], dim=1)

        sa = self.spatial_attention(sa_input)

        x = x * sa

        return x

        class FusionNet(nn.Module):

    def __init__(self):

        super(FusionNet, self).__init__()

        # encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # CBAM
        self.cbam = CBAM(64)

        # decoder
        self.decoder = nn.Sequential(

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, vis, ir):

        x = torch.cat([vis, ir], dim=1)

        x = self.encoder(x)

        x = self.cbam(x)

        fused = self.decoder(x)

        return fused

        # dataset oluştur

train_dataset = FusionDataset(
    visible_train_dir,
    infrared_train_dir,
    train_files
)

val_dataset = FusionDataset(
    visible_train_dir,
    infrared_train_dir,
    val_files
)

# dataloader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

num_epochs = 10

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for vis, ir in train_loader:

        vis = vis.to(device)
        ir = ir.to(device)

        optimizer.zero_grad()

        fused = model(vis, ir)

        loss_vis = criterion(fused, vis)
        loss_ir  = criterion(fused, ir)

        loss = loss_vis + loss_ir

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    model.eval()

with torch.no_grad():

    vis, ir = val_dataset[0]

    vis_input = vis.unsqueeze(0).to(device)
    ir_input  = ir.unsqueeze(0).to(device)

    fused = model(vis_input, ir_input)

    fused = fused.squeeze(0).cpu().permute(1,2,0).numpy()

    vis = vis.permute(1,2,0).numpy()
    ir  = ir.permute(1,2,0).numpy()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(vis)
plt.title("Visible Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(ir)
plt.title("Infrared Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(fused)
plt.title("CBAM Fused Image")
plt.axis("off")

plt.tight_layout()

plt.show()

!pip install pytorch-msssim -q

from pytorch_msssim import ssim

# NEW EXPERIMENT:
# L1 + SSIM LOSS

num_epochs = 10

improved_train_losses = []

# modeli sıfırdan başlat
model = FusionNet().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for vis, ir in train_loader:

        vis = vis.to(device)
        ir = ir.to(device)

        optimizer.zero_grad()

        # fusion output
        fused = model(vis, ir)

        # -------------------------
        # L1 reconstruction losses
        # -------------------------

        loss_vis_l1 = criterion(fused, vis)
        loss_ir_l1  = criterion(fused, ir)

        # -------------------------
        # SSIM structural losses
        # -------------------------

        loss_vis_ssim = 1 - ssim(
            fused,
            vis,
            data_range=1.0
        )

        loss_ir_ssim = 1 - ssim(
            fused,
            ir,
            data_range=1.0
        )

        # -------------------------
        # Total loss
        # -------------------------

        loss = (
            loss_vis_l1
            + loss_ir_l1
            + 0.5 * loss_vis_ssim
            + 0.5 * loss_ir_ssim
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    improved_train_losses.append(epoch_loss)

    print(
        f"[Improved Experiment] "
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f}"
    )

    model.eval()

with torch.no_grad():

    vis, ir = val_dataset[0]

    vis_input = vis.unsqueeze(0).to(device)
    ir_input  = ir.unsqueeze(0).to(device)

    fused = model(vis_input, ir_input)

    fused = fused.squeeze(0).cpu().permute(1,2,0).numpy()

    vis = vis.permute(1,2,0).numpy()
    ir  = ir.permute(1,2,0).numpy()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(vis)
plt.title("Visible Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(ir)
plt.title("Infrared Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(fused)
plt.title("Improved CBAM Fusion (L1 + SSIM)")
plt.axis("off")

plt.tight_layout()

plt.show()

CBAM_DIR = "/content/drive/MyDrive/LLVIP_CBAM_FUSED"

train_cbam_dir = os.path.join(CBAM_DIR, "images", "train")
val_cbam_dir   = os.path.join(CBAM_DIR, "images", "val")

train_lbl_dir = os.path.join(CBAM_DIR, "labels", "train")
val_lbl_dir   = os.path.join(CBAM_DIR, "labels", "val")

for d in [
    train_cbam_dir,
    val_cbam_dir,
    train_lbl_dir,
    val_lbl_dir
]:
    os.makedirs(d, exist_ok=True)

print("CBAM dataset folders created.")

def save_cbam_fused_images(file_list, save_dir):

    model.eval()

    with torch.no_grad():

        for img_file in file_list:

            vis_path = os.path.join(
                visible_train_dir,
                img_file
            )

            ir_path = os.path.join(
                infrared_train_dir,
                img_file
            )

            vis = cv2.imread(vis_path)
            ir  = cv2.imread(ir_path)

            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            ir  = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

            vis = cv2.resize(vis, (256,256))
            ir  = cv2.resize(ir, (256,256))

            vis_tensor = transforms.ToTensor()(vis).unsqueeze(0).to(device)
            ir_tensor  = transforms.ToTensor()(ir).unsqueeze(0).to(device)

            fused = model(vis_tensor, ir_tensor)

            fused = fused.squeeze(0).cpu().permute(1,2,0).numpy()

            fused = (fused * 255).astype(np.uint8)

            fused = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(save_dir, img_file)

            cv2.imwrite(save_path, fused)

    print("Saved:", save_dir)

    save_cbam_fused_images(
    train_files,
    train_cbam_dir
)

save_cbam_fused_images(
    val_files,
    val_cbam_dir
)

import shutil

for img_file in train_files:

    label_file = img_file.replace(".jpg", ".txt")

    src = os.path.join(labels_dir, label_file)

    if os.path.exists(src):

        shutil.copy(
            src,
            os.path.join(train_lbl_dir, label_file)
        )

for img_file in val_files:

    label_file = img_file.replace(".jpg", ".txt")

    src = os.path.join(labels_dir, label_file)

    if os.path.exists(src):

        shutil.copy(
            src,
            os.path.join(val_lbl_dir, label_file)
        )

print("Labels copied.")

yaml_path = os.path.join(CBAM_DIR, "cbam.yaml")

yaml_content = f"""
path: {CBAM_DIR}
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("CBAM YAML created.")

from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")

model_yolo.train(
    data=yaml_path,
    epochs=10,
    imgsz=640,
    batch=8,
    device=0
)

class CBAM_UNet_Fusion(nn.Module):

    def __init__(self):

        super(CBAM_UNet_Fusion, self).__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # -------------------------
        # CBAM
        # -------------------------

        self.cbam = CBAM(128)

        # -------------------------
        # Decoder
        # -------------------------

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        # -------------------------
        # Output
        # -------------------------

        self.out_conv = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, vis, ir):

        # concatenate
        x = torch.cat([vis, ir], dim=1)

        # -------------------------
        # Encoder
        # -------------------------

        e1 = self.enc1(x)

        p1 = self.pool1(e1)

        e2 = self.enc2(p1)

        p2 = self.pool2(e2)

        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(p2)

        b = self.cbam(b)

        # -------------------------
        # Decoder
        # -------------------------

        d1 = self.up1(b)

        d1 = torch.cat([d1, e2], dim=1)

        d1 = self.dec1(d1)

        d2 = self.up2(d1)

        d2 = torch.cat([d2, e1], dim=1)

        d2 = self.dec2(d2)

        # output
        out = self.out_conv(d2)

        out = self.sigmoid(out)

        return out

        model = CBAM_UNet_Fusion().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

print(model)

num_epochs = 10

experiment3_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for vis, ir in train_loader:

        vis = vis.to(device)
        ir = ir.to(device)

        optimizer.zero_grad()

        fused = model(vis, ir)

        # -------------------------
        # L1 losses
        # -------------------------

        loss_vis_l1 = criterion(fused, vis)

        loss_ir_l1 = criterion(fused, ir)

        # -------------------------
        # SSIM losses
        # -------------------------

        loss_vis_ssim = 1 - ssim(
            fused,
            vis,
            data_range=1.0
        )

        loss_ir_ssim = 1 - ssim(
            fused,
            ir,
            data_range=1.0
        )

        # -------------------------
        # total loss
        # -------------------------

        loss = (
            loss_vis_l1
            + loss_ir_l1
            + 0.5 * loss_vis_ssim
            + 0.5 * loss_ir_ssim
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    experiment3_losses.append(epoch_loss)

    print(
        f"[Experiment 3] "
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f}"
    )

    model.eval()

with torch.no_grad():

    vis, ir = val_dataset[0]

    vis_input = vis.unsqueeze(0).to(device)

    ir_input = ir.unsqueeze(0).to(device)

    fused = model(vis_input, ir_input)

    fused = fused.squeeze(0).cpu().permute(1,2,0).numpy()

    vis = vis.permute(1,2,0).numpy()

    ir = ir.permute(1,2,0).numpy()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(vis)
plt.title("Visible Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(ir)
plt.title("Infrared Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(fused)
plt.title("Experiment 3: CBAM-U-Net Fusion")
plt.axis("off")

plt.tight_layout()

plt.show()

# EXPERIMENT 3 MODEL
# CBAM-U-Net + L1 + SSIM

model_exp3 = CBAM_UNet_Fusion().to(device)

optimizer = optim.Adam(
    model_exp3.parameters(),
    lr=1e-4
)

num_epochs = 10
exp3_losses = []

for epoch in range(num_epochs):

    model_exp3.train()
    running_loss = 0.0

    for vis, ir in train_loader:

        vis = vis.to(device)
        ir = ir.to(device)

        optimizer.zero_grad()

        fused = model_exp3(vis, ir)

        loss_vis_l1 = criterion(fused, vis)
        loss_ir_l1 = criterion(fused, ir)

        loss_vis_ssim = 1 - ssim(fused, vis, data_range=1.0)
        loss_ir_ssim = 1 - ssim(fused, ir, data_range=1.0)

        loss = (
            loss_vis_l1
            + loss_ir_l1
            + 0.5 * loss_vis_ssim
            + 0.5 * loss_ir_ssim
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    exp3_losses.append(epoch_loss)

    print(f"[Experiment 3] Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    EXP3_DIR = "/content/drive/MyDrive/LLVIP_EXP3"

train_exp3_dir = os.path.join(EXP3_DIR, "images", "train")
val_exp3_dir = os.path.join(EXP3_DIR, "images", "val")

train_lbl_dir_exp3 = os.path.join(EXP3_DIR, "labels", "train")
val_lbl_dir_exp3 = os.path.join(EXP3_DIR, "labels", "val")

for d in [train_exp3_dir, val_exp3_dir, train_lbl_dir_exp3, val_lbl_dir_exp3]:
    os.makedirs(d, exist_ok=True)

print("Experiment 3 folders created.")

def save_exp3_fused_images(file_list, save_dir):

    model_exp3.eval()

    with torch.no_grad():

        for img_file in file_list:

            vis_path = os.path.join(visible_train_dir, img_file)
            ir_path = os.path.join(infrared_train_dir, img_file)

            vis = cv2.imread(vis_path)
            ir = cv2.imread(ir_path)

            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

            vis = cv2.resize(vis, (256, 256))
            ir = cv2.resize(ir, (256, 256))

            vis_tensor = transforms.ToTensor()(vis).unsqueeze(0).to(device)
            ir_tensor = transforms.ToTensor()(ir).unsqueeze(0).to(device)

            fused = model_exp3(vis_tensor, ir_tensor)

            fused = fused.squeeze(0).cpu().permute(1, 2, 0).numpy()
            fused = (fused * 255).astype(np.uint8)
            fused = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(save_dir, img_file), fused)

    print("Saved:", save_dir)

    

save_exp3_fused_images(train_files, train_exp3_dir)
save_exp3_fused_images(val_files, val_exp3_dir)

import shutil

for img_file in train_files:
    label_file = img_file.replace(".jpg", ".txt")
    src = os.path.join(labels_dir, label_file)

    if os.path.exists(src):
        shutil.copy(src, os.path.join(train_lbl_dir_exp3, label_file))

for img_file in val_files:
    label_file = img_file.replace(".jpg", ".txt")
    src = os.path.join(labels_dir, label_file)

    if os.path.exists(src):
        shutil.copy(src, os.path.join(val_lbl_dir_exp3, label_file))

print("Experiment 3 labels copied.")

yaml_path_exp3 = os.path.join(EXP3_DIR, "exp3.yaml")

yaml_content = f"""
path: {EXP3_DIR}
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path_exp3, "w") as f:
    f.write(yaml_content)

print("Experiment 3 YAML created:", yaml_path_exp3)

from ultralytics import YOLO

model_yolo_exp3 = YOLO("yolov8n.pt")

model_yolo_exp3.train(
    data=yaml_path_exp3,
    epochs=10,
    imgsz=640,
    batch=8,
    device=0
)

import pandas as pd

results_df = pd.DataFrame({

    "Method": [

        "Visible-only",

        "Average Fusion\n(0.5 RGB / 0.5 IR)",

        "Weighted Fusion\n(0.3 RGB / 0.7 IR)",

        "Weighted Fusion\n(0.7 RGB / 0.3 IR)",

        "Experiment 2\n(CBAM + L1 + SSIM)",

        "Experiment 3\n(CBAM-U-Net + L1 + SSIM)"
    ],

    "Precision": [
        0.907,
        0.938,
        0.899,
        0.935,
        0.929,
        0.927
    ],

    "Recall": [
        0.734,
        0.843,
        0.831,
        0.884,
        0.847,
        0.835
    ],

    "mAP50": [
        0.812,
        0.920,
        0.894,
        0.931,
        0.926,
        0.922
    ],

    "mAP50-95": [
        0.401,
        0.531,
        0.498,
        0.555,
        0.535,
        0.529
    ]
})

results_df

import os

PLOT_DIR = "/content/drive/MyDrive/LLVIP_Final_Plots"

os.makedirs(PLOT_DIR, exist_ok=True)

print(PLOT_DIR)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

bars = plt.bar(
    results_df["Method"],
    results_df["mAP50"]
)

plt.title(
    "mAP50 Comparison of Fusion Methods",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("mAP50")

plt.ylim(0.7, 1.0)

plt.xticks(rotation=15)

# değerleri yaz
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.005,
        f"{yval:.3f}",
        ha='center',
        fontsize=10
    )

plt.tight_layout()

save_path = os.path.join(
    PLOT_DIR,
    "mAP50_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import numpy as np

metrics = [
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95"
]

x = np.arange(len(results_df["Method"]))

width = 0.2

plt.figure(figsize=(14,7))

for i, metric in enumerate(metrics):

    plt.bar(
        x + i*width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x + width*1.5,
    results_df["Method"],
    rotation=15
)

plt.ylabel("Score")

plt.ylim(0.3, 1.0)

plt.title(
    "Performance Comparison Across Fusion Experiments",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

save_path = os.path.join(
    PLOT_DIR,
    "all_metrics_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10,6))

plt.plot(
    improved_train_losses,
    label="Experiment 2"
)

plt.plot(
    experiment3_losses,
    label="Experiment 3"
)

plt.title(
    "Fusion Training Loss Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    PLOT_DIR,
    "loss_curve_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
