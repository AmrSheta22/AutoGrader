import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import numpy
import cv2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels=2, out_channels=8, kernel_size=3, stride=2, padding=1
        )  # (2, 2880, 48) ==> (8, 1440, 24)
        self.conv1 = nn.Conv2d(
            in_channels=8, out_channels=30, kernel_size=3, stride=2, padding=1
        )  # (8, 1440, 24) ==> (30, 720, 12)
        self.conv2 = nn.Conv2d(
            in_channels=30, out_channels=60, kernel_size=3, stride=2, padding=1
        )  # (30, 720, 12) ==> (60, 360, 6)
        self.linear = nn.Linear(129600, 1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, img):
        img = self.conv0(img)
        img = self.conv1(img)
        img = self.conv2(img)
        img1 = self.relu(self.flatten(img))
        img1 = img1.view(img1.size(0), -1)  # Flatten the tensor for the linear layer
        out = self.sigm(self.linear(img1))
        return out, img


def split_image(image):
    width, height = image.shape[1], image.shape[0]
    mid_point = width // 2
    img1 = image[0:height, 0:mid_point]
    img2 = image[0:height, mid_point:width]
    # img1_path = os.path.splitext(image_path)[0] + '_part1.png'
    # img2_path = os.path.splitext(image_path)[0] + '_part2.png'
    # img1.save(img1_path)
    # img2.save(img2_path)
    return img1, img2


def process_and_predict(img1, img2, model, device, transform):
    
    img1 = Image.fromarray(np.uint8(img1))
    img2 = Image.fromarray(np.uint8(img2))
    img1 = transform(img1)
    img2 = transform(img2)
    img = torch.cat((img1, img2), dim=0).unsqueeze(0)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        prediction, conv2_layer = model(img)
        conv2_layer = conv2_layer.view(1, -1, 360, 360)
        padding = (12, 12, 12, 12)
        padded_embedding = F.pad(conv2_layer, padding, mode="constant", value=0)
        embedding = padded_embedding.squeeze().view(1, 384, 384)
    return embedding, prediction


class CustomTrOCR(nn.Module):
    def __init__(self):
        super(CustomTrOCR, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, image_input, vector_input):
        concatenated_input = torch.cat((image_input, vector_input), dim=1)
        x = self.conv(concatenated_input)
        return x

    def generate(self, image_input, vector_input):
        concatenated_input = torch.cat((image_input, vector_input), dim=0)
        x = self.conv(concatenated_input)
        return x


class MergedTrOCR(nn.Module):
    def __init__(self, ocr_model):
        super(MergedTrOCR, self).__init__()
        self.additional_layer = CustomTrOCR()
        self.model = ocr_model

    def forward(self, image_input, vector_input, decoder_input_ids):
        image_input = self.additional_layer(image_input, vector_input)
        x = self.model(image_input, decoder_input_ids=decoder_input_ids)
        return x

    def generate(self, image_input, vector_input):
        image_input = self.additional_layer(image_input, vector_input)
        x = self.model.generate(image_input, num_beams=5)
        return x


def binarization_2(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    return img_bin


def noise_removal_3(image):
    kernel = np.ones((1, 1), np.uint8)
    no_noise_image = cv2.dilate(image, kernel, iterations=1)
    no_noise_image = cv2.erode(no_noise_image, kernel, iterations=1)
    no_noise_image = cv2.morphologyEx(no_noise_image, cv2.MORPH_CLOSE, kernel)
    no_noise_image = cv2.medianBlur(no_noise_image, 3)
    return no_noise_image


def thick_font_4(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    dilated_image = cv2.bitwise_not(dilated_image)
    return dilated_image


class Processor:
    def __init__(self, processor, state_dict, device):
        self.processor = processor
        self.device = device
        self.max_target_length = 128
        self.SiameseModel = SiameseNetwork()
        self.SiameseModel_path = "siamese_model.pth"
        self.state_dict = torch.load(
            self.SiameseModel_path, map_location=torch.device("cpu")
        )
        self.SiameseModel.load_state_dict(state_dict)
        self.SiameseModel.eval()
        self.transform_embedding = transforms.Compose(
            [
                transforms.Resize((2880, 48)),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ]
        )

    def process(self, image):
        img1, img2 = split_image(image)
        SiameseModel = self.SiameseModel.to(self.device)
        embedding, prediction = process_and_predict(
            img1, img2, SiameseModel, self.device, self.transform_embedding
        )
        # image = Image.open(img_path).convert('RGB')
        image = numpy.array(image)
        image = binarization_2(image)
        image = noise_removal_3(image)
        image = thick_font_4(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        return pixel_values, embedding


import cv2
import numpy as np


def contains_black_pixels(segment, threshold=10000):
    segment = np.abs(255 - segment)
    total_sum = np.sum(segment)            
    return total_sum > threshold

def image_to_segments(image, width=800):
    image = cv2.resize(image, (800, int(800 * image.shape[0] / image.shape[1])))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 400)
    if lines is None or len(lines) == 0:
        height, width = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count_top = 0
        count_left = 0
        count_right = 0
        for i in range(height):
            if sum(img_gray[i])<1000:
                count_top +=1
            if sum(img_gray[:,i])<1000:
                count_left +=1
            if sum(img_gray[:,-i])<1000:
                count_right +=1
        if height > 5:
            trimmed_segment = image[count_top:height, count_left:width - count_right-1]
        thresholded_arr = (np.abs(trimmed_segment[:,:,1]-255) > 127).astype(int)*255
        total_sum = np.sum(thresholded_arr)
        print(total_sum)
        if total_sum < 100000:
            return []
        else:
            return [image]
    lines.sort(axis=0)
    images_segments = []
    for i, line in enumerate(lines):
        if lines[i][0][0] - lines[i - 1][0][0] > width / 40:
            images_segments.append(image[int(lines[i - 1][0][0]) : int(lines[i][0][0])])
    if len(images_segments) == 0:
        if len(lines) == 1:
            #x = lines[0][0][0]
            height, width = image.shape[:2]
            x = int(np.ceil(lines[0][0][1]))
            image = image[x:height]
            print(image.shape)
            images_segments.append(image)
        else:
            images_segments.append(image)
    trimmed_segments = []
    for segment in images_segments:

        count_top = 0
        count_left = 0
        count_right = 0
        height, width = segment.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i in range(height):
            if sum(img_gray[i])<1000:
                count_top +=1
            if sum(img_gray[:,i])<1000:
                count_left +=1
            if sum(img_gray[:,-i])<1000:
                count_right +=1

        if height > 5:
            trimmed_segment = segment[count_top:height, count_left:width - count_right-1]
            img_gray = img_gray[0:height+count_top, count_left:width - count_right-1]
        thresholded_arr = (np.abs(trimmed_segment[:,:,1]-255) > 127).astype(int)*255
        total_sum = np.sum(thresholded_arr)
        print(total_sum)
        if total_sum < 100000:
            continue
        else:
            removed_white= 0
            threshold = 125
            thresholded_arr = (np.abs(trimmed_segment[:,:,1]-255) > threshold).astype(int)*255
            sum_y = np.sum(thresholded_arr, axis=0)
            for i in range(img_gray.shape[1]):
                if sum_y[-i] < 800:
                    removed_white+=1
                else:
                    break
            print(removed_white)
            trimmed_segment = trimmed_segment[:, 0:width-removed_white+1, :]
            trimmed_segments.append(trimmed_segment)
    return trimmed_segments


def initialize_model(
    SiameseModel_path="siamese_model.pth",
    main_model_path="TrOCR-finetuned-iamcvl/TrOCR-finetuned.pt",
):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    ocr_model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )
    state_dict = torch.load(SiameseModel_path, map_location=torch.device("cpu"))
    SiameseModel = SiameseNetwork()
    SiameseModel.load_state_dict(state_dict)
    SiameseModel.eval()
    model = MergedTrOCR(ocr_model)
    model = torch.load(main_model_path, map_location=torch.device('cpu'))
    model = model.to(device)
    ps = Processor(processor, state_dict, device)
    return ps, model, device, processor


def inference_model(ps, model, processor, device, img):
    images, vector_input = ps.process(img)
    images = images.to(device).unsqueeze(0)
    vector_input = vector_input.to(device).unsqueeze(0)
    ex_generate = model.generate(images, vector_input)
    ocr_output = processor.batch_decode(ex_generate, skip_special_tokens=True)[0]
    return ocr_output


def httr_prediction(ps, model, processor, device, img):
    segments = image_to_segments(img)
    if len(segments)==0:
        return "0"
    output = []
    for segment in segments:
        output.append(inference_model(ps, model,processor, device, segment))
    out = " ".join(output)
    if out == "":
        return " "
    else:
        return out