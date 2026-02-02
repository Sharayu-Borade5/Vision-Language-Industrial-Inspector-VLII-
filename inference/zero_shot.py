import torch
import torchvision.transforms as T
from PIL import Image

def zero_shot_predict(model, image_path, text_tokens):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        sim = model(img, text_tokens)
    return sim.softmax(dim=-1)
