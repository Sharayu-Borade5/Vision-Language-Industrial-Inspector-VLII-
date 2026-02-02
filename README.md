# Vision-Language Industrial Inspector (VLII)

Zero-shot and few-shot industrial defect detection using vision–language alignment.

## 🔍 Overview
VLII uses a CLIP-style architecture to detect manufacturing defects
by matching images against natural language descriptions such as:
- "scratched metal surface"
- "missing screw"
- "cracked component"

No task-specific classifier is required.

## ✨ Features
- Zero-shot defect detection
- Few-shot fine-tuning
- Vision Transformer / ResNet backbones
- Text prompt engineering
- Explainability with Grad-CAM
- Config-driven pipeline

## 🧠 Architecture
Image Encoder → Projection → Shared Embedding Space ← Projection ← Text Encoder

## 🧪 Tasks
- Industrial surface inspection
- Quality control
- Unknown defect discovery

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python demo.py --image data/sample_images/test.jpg
