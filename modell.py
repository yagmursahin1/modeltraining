import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from IPython.display import Image, display

from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.models import YOLO
from ultralytics.utils.loss import v8DetectionLoss

class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module"""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class BiFPNLayer(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon
        self.channels = channels

        # Learnable weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        # Separable convolutions
        self.conv_up = self._make_separable_conv(channels, channels)
        self.conv_down = self._make_separable_conv(channels, channels)

    def _make_separable_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, features):
        P3, P4, P5 = features

        # Top-down path
        w1 = F.relu(self.w1)
        P5_up = F.interpolate(P5, size=P4.shape[2:], mode='nearest')
        P4_td = (w1[0] * P4 + w1[1] * P5_up) / (w1[0] + w1[1] + self.epsilon)
        P4_td = self.conv_up(P4_td)

        P4_up = F.interpolate(P4_td, size=P3.shape[2:], mode='nearest')
        P3_out = (w1[0] * P3 + w1[1] * P4_up) / (w1[0] + w1[1] + self.epsilon)
        P3_out = self.conv_up(P3_out)

        # Bottom-up path
        w2 = F.relu(self.w2)
        P3_down = F.max_pool2d(P3_out, kernel_size=2, stride=2)
        P4_out = (w2[0] * P4 + w2[1] * P4_td + w2[2] * P3_down) / (w2.sum() + self.epsilon)
        P4_out = self.conv_down(P4_out)

        P4_down = F.max_pool2d(P4_out, kernel_size=2, stride=2)
        P5_out = (w1[0] * P5 + w1[1] * P4_down) / (w1[0] + w1[1] + self.epsilon)
        P5_out = self.conv_down(P5_out)

        return [P3_out, P4_out, P5_out]
    
class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2):
        super(BiFPN, self).__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(channels) for _ in range(num_layers)
        ])

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features
    

class WIoULoss(nn.Module):
    def __init__(self, monotonous=False):
        super(WIoULoss, self).__init__()
        self.monotonous = monotonous

    def forward(self, pred, target, eps=1e-7):
        # IoU calculation
        inter_area = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])).clamp(0) * \
                    (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1])).clamp(0)

        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + eps)

        # Wise IoU calculation
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)

        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = enclose_x2y2 - enclose_x1y1
        diagonal = torch.sum(enclose_wh ** 2, dim=1)

        r = center_distance / (diagonal + eps)

        if self.monotonous:
            beta = r / (1 - iou + eps)
            loss = -torch.log(iou + eps) * torch.exp(beta)
        else:
            beta = r * r / ((1 - iou) ** 2 + eps)
            loss = 1 - iou * torch.exp(-beta)

        return loss.mean()

class EnhancedNeck(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=512):
        super(EnhancedNeck, self).__init__()

        # Channel adjustment
        self.reduce_layers = nn.ModuleList([
            Conv(ch, out_channels, 1, 1) for ch in in_channels
        ])

        # SimAM attention
        self.simam_layers = nn.ModuleList([
            SimAM() for _ in range(3)
        ])

        # BiFPN
        self.bifpn = BiFPN(out_channels, num_layers=2)

        # Output processing
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                C2f(out_channels, out_channels, n=1),
                Conv(out_channels, out_channels, 3, 1)
            ) for _ in range(3)
        ])

    def forward(self, x):
        # Channel reduction ve SimAM
        features = []
        for i, (feat, reduce_layer, simam) in enumerate(zip(x, self.reduce_layers, self.simam_layers)):
            feat = reduce_layer(feat)
            feat = simam(feat)
            features.append(feat)

        # BiFPN fusion
        features = self.bifpn(features)

        # Final processing
        outputs = []
        for feat, output_layer in zip(features, self.output_layers):
            outputs.append(output_layer(feat))

        return outputs
    
class EnhancedDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.wiou_loss = WIoULoss()

    def __call__(self, preds, batch):
        # Original loss calculation
        loss = super().__call__(preds, batch)

        # WIoU enhancement can be added here if needed
        # For simplicity, using original loss structure
        return loss


def create_enhanced_model():
    """Enhanced YOLOv8x model oluştur"""

    # Base model yükle
    model = YOLO('yolov8x.pt')

    # Model architecture'ı modifiye et
    def enhance_model(model):
        # Backbone çıkışlarını al
        backbone_channels = [512, 1024, 1024]  # YOLOv8x için

        # Enhanced neck ile değiştir
        enhanced_neck = EnhancedNeck(backbone_channels, 512)

        # Model'in neck kısmını değiştir
        model.model.model[15] = enhanced_neck  # Neck layer index

        # Loss function'ı değiştir
        model.model.loss = EnhancedDetectionLoss(model.model)

        return model

    # Model'i enhance et
    enhanced_model = enhance_model(model)

    return enhanced_model

def train_enhanced_yolo(data_yaml, epochs=300, imgsz=640, batch=16):
    """Enhanced YOLO training"""

    print("Enhanced YOLO model oluşturuluyor...")
    model = create_enhanced_model()

    # Training hyperparameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Enhanced box loss weight
        'cls': 0.5,
        'dfl': 1.5,
        'patience': 100,
        'save': True,
        'save_period': 10,
        'cache': False,  # Colab memory için
        #'device': 0 if torch.cuda.is_available() else 'cpu',
        'device': 'cpu',
        'workers': 2,  # Colab için düşük
        'project': '/content/runs/train',
        'name': 'enhanced_yolo',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Mixed precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
    }

    print("Training başlıyor...")
    print(f"Epochs: {epochs}, Image Size: {imgsz}, Batch Size: {batch}")

    # Training başlat
    results = model.train(**train_args)

    return model, results

def evaluate_model(model, data_yaml):
    """Model değerlendirmesi"""

    print("Model değerlendiriliyor...")

    # Validation
    val_results = model.val(data=data_yaml, imgsz=640, batch=16, conf=0.001, iou=0.6)

    print(f"Validation Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    # Test prediction
    test_results = model('/content/coco/images/val2017', save=True, conf=0.25)

    return val_results

def visualize_results():
    """Sonuçları görselleştir"""

    # Training curves
    training_results_path = '/content/runs/train/enhanced_yolo/results.png'
    if os.path.exists(training_results_path):
        display(Image(training_results_path))

    # Prediction samples
    pred_path = '/content/runs/detect/predict'
    if os.path.exists(pred_path):
        pred_images = list(Path(pred_path).glob('*.jpg'))[:3]  # İlk 3 image
        for img_path in pred_images:
            print(f"Prediction: {img_path}")
            display(Image(str(img_path)))

def main_custom_dataset(dataset_path=None, yaml_path=None, epochs=300, batch=16, imgsz=640):

      print(" Enhanced YOLOv8 Custom Dataset Training Başlıyor...")
      print("="*60)

      # 1. Dataset setup
      print(" Custom dataset setup ediliyor...")

      if yaml_path:
          # YAML dosyası varsa direkt kullan
          data_yaml = "/content/drive/MyDrive/IHA3/data-2.yaml"
      else:
          print("❌ data.yaml dosyası belirtilmedi!")
          return None
      
def main():
    """Ana training pipeline"""

    print("Enhanced YOLOv8 Training Pipeline Başlıyor...")
    print("="*60)

    # 1. Dataset hazırla
    print(" Dataset hazırlanıyor...")

    data_yaml= "/content/drive/MyDrive/IHA3/data-2.yaml"

    # 2. Model training
    print(" Model training...")
    model, results = train_enhanced_yolo(
        data_yaml= data_yaml,
        epochs=300,  # Colab için uygun
        imgsz=640,
        batch=16   
    )

    # 3. Evaluation

    print(" Model değerlendirmesi...")
    val_results = evaluate_model(model, data_yaml)

    # 4. Visualization
    print(" Sonuçları görselleştir...")
    visualize_results()

    # 5. Model kaydet
    print(" Model kaydediliyor...")
    model.save('/content/enhanced_yolov8x.pt')

    print(" Training tamamlandı!")
    print(f"Model kaydedildi: /content/enhanced_yolov8x.pt")

    return model

main()
