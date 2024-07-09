from PIL import Image

def show_image(pathStr):
    img = Image.open(pathStr).convert("RGB")
    display(img)

def TrOCR_image(path):
    # Load the image
    image = Image.open(path).convert("RGB")

    # Process the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate OCR output
    pixel_values = pixel_values.to(device)
    generated_ids = model.generate(pixel_values, num_beams=5)
    
    ocr_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Remove non-alphanumeric characters and replace multiple spaces with a single space
    ocr_output_cleaned = re.sub(r'[^a-zA-Z0-9\s\'-]', ' ', ocr_output)
    ocr_output_cleaned = ' '.join(ocr_output_cleaned.split())

    # Remove hyphens surrounded by spaces
    ocr_output_cleaned = re.sub(r'\s-\s', ' ', ocr_output_cleaned)

    return ocr_output_cleaned
