import os
from scr.preprocessing_functions import *

all_data = os.listdir("IAM_examples")
height, width = get_max_dimensions(all_data)
for i in all_data:
    img = binarization(img)
    img = noise_removal(img)
    img = thick_font(img)
    img = deskew(img)
    img = deslant_image(img)
    img = rescaling(img, height, width, color=(255, 255, 255))
    output = "preprocessed_examples" + i
    cv2.imwrite(output, img)
