import torch
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.vlii_model import VLII
from inference.zero_shot import zero_shot_predict

# Example defect prompts (dummy tokens for now)
text_tokens = torch.randint(0, 10000, (4, 10))

vision = VisionEncoder(512)
text = TextEncoder()
model = VLII(vision, text)
model.eval()

scores = zero_shot_predict(
    model,
    "data/sample_images/test.jpg",
    text_tokens
)

print("Zero-shot probabilities:", scores)
