import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def Pil2cv_converter(File):
    """
    This Function takes PIL file And converts it to opencv file

    Parameters
    
    File : The PIL file
    """
    # Open a PIL image
    pil_image = File

    # Convert PIL image to NumPy array
    numpy_array = np.array(pil_image)

    # Convert NumPy array to OpenCV image
    # note native OpenCV files are BGR you can convert to RGB later
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
    return opencv_image