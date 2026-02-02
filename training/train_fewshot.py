import torch
from models.vlii_model import VLII
from training.losses import contrastive_loss

def train(model, loader, optimizer):
    model.train()
    for images, tokens in loader:
        sim = model(images, tokens)
        loss = contrastive_loss(sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
