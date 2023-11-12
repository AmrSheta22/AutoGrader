# the imports
import os
import matplotlib.pyplot as plt
import cv2
from deslant_img import deslant_img
from wand.image import Image
from wand.display import display
from matplotlib import cm

os.environ["MAGIK_HOME"] = "C:\Program Files\ImageMagick-7.1.1-Q16-HDRI"
import numpy as np


def Read_IAM(txt_path, data_location):
    """
    this function reads the IAM data sets through its Text File and outputs the paths to the images files and other information which are numebr of
    component, gray scale level for binarization , location of bounding boxes in the image data set, state of it image in terms of segmentation
    (ok: line is correctly segmented ,err: segmentation of line has one or more errors) and its true label
    the order of the returned objects are as follows: txt_path , txt_state , txt_label ,txt_graylevel, txt_number_of_components,txt_bounding_box

    txt_path : the txt_path to the text file in IAM line data set
    data_location : the location of the data in your machine for example "D:\Academics\DS\Project\data\Data\IAM\A The og\the Try"
    """
    with open(txt_path, "r") as file:
        txt = file.read()
    txt = txt.splitlines()
    txt_1 = [i for i in txt if not i.startswith("#")]
    txt_path = []
    txt_state = []
    txt_label = []
    txt_graylevel = []
    txt_number_of_components = []
    txt_bounding_box = []
    for i in txt_1:
        txt_path.append(i.split()[0])
        txt_label.append(i.split()[-1])
        txt_number_of_components.append(i.split()[3])
        txt_state.append(i.split()[1])
        txt_graylevel.append(i.split()[2])
        txt_bounding_box.append(i.split()[4:8])
    path_t = []
    for i in range(len(txt_path)):
        path_t.append(
            r""
            + os.path.join(
                r"" + data_location,
                txt_path[i].split("-")[0],
                txt_path[i].split("-")[0] + "-" + txt_path[i].split("-")[1],
                txt_path[i] + ".png",
            )
        )
    return (
        path_t,
        txt_label,
        txt_graylevel,
        txt_number_of_components,
        txt_bounding_box,
    )


# not to all who uses this code this work if you download the data form the official website (the link in the Documentation) because it uses
# the location of files after unzipping the data
def Rimes_read(txt_path_1, txt_path_2, txt_path_3, data_location):
    """
    this function reads the IAM data sets through its Text File and outputs the paths to the images files and other information which are path and
    labels

    note in Rimes data sets there is 3 text files

    txt_path_1 : the txt_path to the text file in IAM line data set
    txt_path_2 : the txt_path to the text file in IAM line data set
    txt_path_3 : the txt_path to the text file in IAM line data set
    data_location : the location of the data in your machine for example "D:\Academics\DS\Project\data\Data\koimes"
    """
    with open(r"" + txt_path_1, "r") as file:
        text = file.read()
    text_1 = text.splitlines()

    with open(r"" + txt_path_2, "r") as file:
        text2 = file.read()

    text2_2 = text2.splitlines()

    with open(r"" + txt_path_3, "r") as file:
        text3 = file.read()
    text3_3 = text3.splitlines()
    text_1.extend(text2_2)
    text_1.extend(text3_3)
    paths = []
    trans = []
    the_extracte_name = "RIMES-2011-Lines"
    the_image_file = "Images"
    the_transcript_file = "Transcriptions"
    for i in range(len(text_1)):
        jojo = r"" + os.path.join(
            r"" + data_location,
            r"" + the_extracte_name,
            r"" + the_image_file,
            text_1[i] + ".jpg",
        )
        paths.append(jojo)
    for i in range(len(text_1)):
        jojo1 = r"" + os.path.join(
            r"" + data_location,
            r"" + the_extracte_name,
            r"" + the_transcript_file,
            text_1[i] + ".txt",
        )
        with open(jojo1, "r") as file:
            bobo = file.read()
        trans.append(bobo)
    return paths, trans


# this method is used to find the max dimensions of our datasets
# this function loop on the file of images to det the max dimensions on it
def get_max_dimensions(paths):
    # intial the height and width
    max_height = 0
    max_width = 0
    for path in paths:
        if path.endswith(".jpg") or path.endswith(".png"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
            # get the dimentions of the image to save in the variables height and weight
            height, width = img.shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)
    return max_height, max_width


def invert(img):
    img = cv2.bitwise_not(img)
    return img

def binarization(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)[1]
    return img_bin

def noise_removal(img):
    kernel = np.ones((1, 1), np.uint8)
    no_nois_image = cv2.dilate(img, kernel, iterations=1)
	#kernel = np.ones((1, 1), np.uint8)
    no_nois_image = cv2.erode(no_nois_image, kernel, iterations=1)
    no_nois_image = cv2.morphologyEx(no_nois_image, cv2.MORPH_CLOSE, kernel)
    no_nois_image = cv2.medianBlur(no_nois_image, 3)  
    return(no_nois_image)

#Dilation and Erosion
def thick_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    dilated_image = cv2.dilate(img, kernel, iterations=1)
    dilated_image = cv2.bitwise_not(dilated_image)
    return(dilated_image)

def deskew(img):
    #img = cv2.medianBlur(img, 3)
    img=Image.from_array(img)
    img.deskew(0.4*img.quantum_range)
    #display(img)
    image = np.array(img)
    image = np.squeeze(image)
    return image

def deslant_image(img):
	#Need to do preprocessing because deslant_img need the image 2D
	res = deslant_img(img,bg_color=255).img
	return res 


def rescaling(img,max_height,max_width,color=(255, 255, 255)):
	#apply preprocess_line_image()function to make the image 2D
	# Get the height and width of the current image
	img_height, img_width = img.shape
	# Calculate the resize ratio based on max_height divided by the current image's height
	ratio = max_height / img_height
	# Resize the image
	resized_img = img
	# Calculate padding values based on the resized image's dimensions
	pad_height = max(0, max_height - resized_img.shape[0])
	pad_width = max(0, max_width - resized_img.shape[1])

	# Add padding to the image only if it's larger than the original
	if pad_height > 0 or pad_width > 0:
		padded_img = cv2.copyMakeBorder(resized_img, pad_height,0, 0, pad_width, cv2.BORDER_CONSTANT, value=color)
	else:
		# If no padding is needed, use the resized image directly
		padded_img = resized_img
	imggg=padded_img
	return imggg

def get_max_dimensions(paths):
	#intial the height and width
    max_height = 0
    max_width = 0
    for path in paths:
        if path.endswith(".jpg") or path.endswith(".png"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
			#get the dimentions of the image to save in the variables height and weight
            height, width = img.shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)
    return max_height, max_width
