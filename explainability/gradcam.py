import torch
import cv2
import numpy as np

def gradcam(feature_map, gradients):
    weights = gradients.mean(axis=(2,3))
    cam = (weights[:, :, None, None] * feature_map).sum(1)
    cam = np.maximum(cam, 0)
    cam /= cam.max()
    return cam
