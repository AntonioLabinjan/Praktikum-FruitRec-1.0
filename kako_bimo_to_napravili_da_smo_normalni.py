import torch
from PIL import Image, ImageDraw
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xclip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

logger = logging.getLogger()

def convert_image_to_tensor(image_path, frame_size=(224, 224)):
    logger.info("Converting image to tensor...")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(frame_size, Image.BILINEAR)
    
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  
    logger.info("Image conversion to tensor complete.")
    
    return image_tensor, image

def classify_fruit(image_tensor, xclip_model, processor, device):
    logger.info("Classifying fruit...")

    fruit_classes = ["apple", "banana", "cherry", "grape", "orange", "strawberry", "watermelon"]
    
    inputs = processor(text=fruit_classes, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = xclip_model(pixel_values=image_tensor, **inputs)
        logits = outputs.logits_per_text

        predictions = torch.argmax(logits, dim=0)
        predicted_class = fruit_classes[predictions.item()]
        logger.info(f"Predicted fruit: {predicted_class}")
    
    logger.info("Fruit classification complete.")
    return predicted_class

def annotate_image(image, predicted_class, output_path):
    logger.info(f"Annotating image with predicted class: {predicted_class}...")

    draw = ImageDraw.Draw(image)
    
    text = f"Predicted: {predicted_class}"
    
    text_position = (10, 10)
    draw.text(text_position, text, fill="black")

    image.save(output_path)
    logger.info(f"Annotated image saved to {output_path}")
  
image_path = '/content/Image_30.jpg'  # Replace with your image path
image_tensor, image = convert_image_to_tensor(image_path)
predicted_fruit = classify_fruit(image_tensor, xclip_model, processor, device)

output_image_path = '/content/annotated_fruit.jpg'  # Replace with desired output path
annotate_image(image, predicted_fruit, output_image_path)

print(f"Predicted Fruit: {predicted_fruit}")
