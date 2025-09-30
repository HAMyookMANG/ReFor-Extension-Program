import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==========================
# Residualizer 모듈
# ==========================
class Residualizer(nn.Module):
    def __init__(self, ksize=5, sigma=1.2, eps=1e-6):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.eps = eps
        ax = torch.arange(ksize) - (ksize - 1) / 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        self.register_buffer("weight", kernel.view(1, 1, ksize, ksize))

    def forward(self, x):
        y = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        y_blur = F.conv2d(y, self.weight, padding=self.ksize // 2)
        r = y - y_blur
        mean = r.mean(dim=[1,2,3], keepdim=True)
        std  = r.std(dim=[1,2,3], keepdim=True)
        r = (r - mean) / (std + self.eps)
        return r.repeat(1, 3, 1, 1)

# ==========================
# 원래 분류기
# ==========================
class ResidualAttribution(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ==========================
# Residualizer + Classifier
# ==========================
class FullModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.residualizer = Residualizer()
        self.classifier = ResidualAttribution(num_classes)

    def forward(self, x):
        r = self.residualizer(x)
        return self.classifier(r)

# ==========================
# 변환 실행
# ==========================
def convert(pth_path, onnx_path, num_classes):
    model = FullModel(num_classes)
    state = torch.load(pth_path, map_location="cpu")
    model.classifier.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )
    print(f"{onnx_path} completed")

if __name__ == "__main__":
    # Binary classifier (2 classes)
    convert("bumhoya_binary_best.pth", "binary_full.onnx", num_classes=2)


