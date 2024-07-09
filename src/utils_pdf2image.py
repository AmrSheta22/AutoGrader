# first is the imports 
import fitz # note in install this use pip install PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
def Pdf2Image(inputtype, path= None , file=None,  quality = 4):
    """
    Pdf2Image function take pdf file and return image/s file in PIL library format

    Parameters
    inputtype : is the input type 0 if path 1 if object note the object type must be fitz
    path : the path of the pdf object to be converted note input this or file note both
    file : the file in fitz library format note input this or path note both
    quality : the quality of the resulting image/s note the higher the quality the higher the size of the image/s defult 4

    """
    if inputtype is None:
        print("please insert Input type 0 if path 1 if file note the object type must be fitz")
        return None

    if inputtype==0:
        docs = fitz.open(r""+path)
        mat = fitz.Matrix(quality,quality)
        pixs = []
        for page in docs:
            pix = page.get_pixmap(matrix =mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pixs.append(img)
        return pixs
    elif inputtype==1:
        docs = file
        mat = fitz.Matrix(quality,quality)
        pixs = []
        for page in docs:
            pix = page.get_pixmap(matrix =mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pixs.append(img)
        return pixs        

