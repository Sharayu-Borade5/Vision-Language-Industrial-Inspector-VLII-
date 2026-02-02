import torch
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.vlii_model import VLII

# Dummy tokens for demo
text_tokens = torch.randint(0, 10000, (5, 10))

vision = VisionEncoder(512)
text = TextEncoder()
model = VLII(vision, text)

scores = model(torch.randn(1,3,224,224), text_tokens)
print("Similarity scores:", scores)
