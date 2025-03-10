from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained("yaya36095/ai-source-detector")
model = ViTForImageClassification.from_pretrained("yaya36095/ai-source-detector")

def detect_code_source(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.softmax(dim=-1)
    prediction_id = torch.argmax(predictions).item()
    confidence = predictions[0][prediction_id].item() * 100
    sources = ["Human", "Midjourney", "DALL-E", "Stable Diffusion"]
    result = sources[prediction_id]
    return result, confidence

result, confidence = detect_code_source("path/to/code_screenshot.jpg")
print(f"Source: {result} (Confidence: {confidence:.2f}%)")
