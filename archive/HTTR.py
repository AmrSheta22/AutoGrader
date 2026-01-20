from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_to_segments(image, width = 800):
    #image = cv2.imread(image_path)
    image = cv2.resize(image, (800, int(800 * image.shape[0] / image.shape[1])))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 400)
    if lines is None or len(lines) == 0:
        return [image]
    lines.sort(axis=0)
    images_segments = []
    for i, line in enumerate(lines):
        if lines[i][0][0] - lines[i-1][0][0] > width/40:
            images_segments.append(image[int(lines[i-1][0][0]):int(lines[i][0][0])])
    if len(images_segments) == 0:
        images_segments.append(image)
    return images_segments

def httr_prediction(model, processor, image):
    #image = Image.open(path).convert("RGB")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    segments = image_to_segments(image, width = 800)
    out_text = ""
    for image in segments:
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values, num_beams=5)
        ocr_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ocr_output_cleaned = re.sub(r'[^a-zA-Z0-9\s\'-]', ' ', ocr_output)
        ocr_output_cleaned = ' '.join(ocr_output_cleaned.split())
        ocr_output_cleaned = re.sub(r'\s-\s', ' ', ocr_output_cleaned)
        out_text += ocr_output_cleaned + " "
    return out_text