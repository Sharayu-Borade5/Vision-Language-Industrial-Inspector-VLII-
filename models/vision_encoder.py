import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.proj(x)
