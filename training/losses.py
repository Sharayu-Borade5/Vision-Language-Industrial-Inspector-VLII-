import torch
import torch.nn.functional as F

def contrastive_loss(similarity, temperature=0.07):
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    similarity /= temperature
    return (F.cross_entropy(similarity, labels) +
            F.cross_entropy(similarity.T, labels)) / 2
