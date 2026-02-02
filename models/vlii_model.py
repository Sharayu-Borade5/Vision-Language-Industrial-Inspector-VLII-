import torch
import torch.nn as nn
import torch.nn.functional as F

class VLII(nn.Module):
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision = vision_encoder
        self.text = text_encoder

    def forward(self, images, text_tokens):
        img_emb = F.normalize(self.vision(images), dim=-1)
        txt_emb = F.normalize(self.text(text_tokens), dim=-1)
        return img_emb @ txt_emb.T
