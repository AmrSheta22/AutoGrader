import cv2
import matplotlib.pyplot as plt
import numpy as np
def computer_vision_soft_version(file):
    """
    This function takes a AG test photo and returns the student answer box

    Parameters
    file :  the Photo of the AG test in opencv file object note high quality as possible
    """
    
    img = file
    imgr = img.copy()
    # convert it to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # binarzation
    img_bin = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)[1]
    # invert the image
    img_bininv = cv2.bitwise_not(img_bin)
    # Apply morphological operations to remove small noise regions Note use this if the image low quality only
    # Note use this if the image low quality only and if there is a error on the number of answers otherwise leave commented out
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #img_bininv = cv2.morphologyEx(img_bininv, cv2.MORPH_OPEN, kernel)

    # getting the contours
    contours , hier = cv2.findContours(img_bininv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Step one of filtering the contours: Geting only the contours with respect hight and width
    good_contours = []
    for con in contours:
        x, y, w, h = cv2.boundingRect(con)
        if h > 70 and w > 700 :
            good_contours.append(con)
    # Step two of filtering the contours: Removing duplicates 
    pre_x = None
    pre_y = None
    good_contourss = []
    for con in good_contours:
        x, y, w, h = cv2.boundingRect(con)
        if h > 70 and w > 700 :
            if pre_x is None and pre_y is None:
                good_contourss.append(con) #print("abroved")
            else:
                if ((abs(x - pre_x) > 300) or (abs(y - pre_y) > 300)):
                    good_contourss.append(con)#print("abroved")
        pre_x = x
        pre_y = y
    # now get each answer in the photo
    Answers = []
    for contour in good_contourss:
        # getting the answer dimensions 
        x, y, w, h = cv2.boundingRect(contour)
        # cropping based on the dimensions
        cropped_image = imgr[y:y+h, x:x+w]
        Answers.append(cropped_image)
    height, width, channels = imgr.shape
    return Answers , good_contourss , height, width
def computer_vision_soft_version_file_ide(files):
    """
    This function takes a AG test photo and returns the student answer box

    Parameters
    file :  the Photo of the AG test in opencv file object note high quality as possible
    """
    loaded_file =[]
    for file in files:
        img = file
        imgr = img.copy()
        # convert it to grayscale
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # binarzation
        img_bin = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)[1]
        # invert the image
        img_bininv = cv2.bitwise_not(img_bin)
        # Apply morphological operations to remove small noise regions Note use this if the image low quality only
        # Note use this if the image low quality only and if there is a error on the number of answers otherwise leave commented out
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #img_bininv = cv2.morphologyEx(img_bininv, cv2.MORPH_OPEN, kernel)

        # getting the contours
        contours , hier = cv2.findContours(img_bininv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Step one of filtering the contours: Geting only the contours with respect hight and width
        good_contours = []
        for con in contours:
            x, y, w, h = cv2.boundingRect(con)
            if h > 70 and w > 700 :
                good_contours.append(con)
        # Step two of filtering the contours: Removing duplicates 
        pre_x = None
        pre_y = None
        good_contourss = []
        for con in good_contours:
            x, y, w, h = cv2.boundingRect(con)
            if h > 70 and w > 700 :
                if pre_x is None and pre_y is None:
                    good_contourss.append(con) #print("abroved")
                else:
                    if ((abs(x - pre_x) > 300) or (abs(y - pre_y) > 300)):
                        good_contourss.append(con)#print("abroved")
            pre_x = x
            pre_y = y
        # now get each answer in the photo
        Answers = []
        for contour in good_contourss:
            # getting the answer dimensions 
            x, y, w, h = cv2.boundingRect(contour)
            # cropping based on the dimensions
            cropped_image = imgr[y:y+h, x:x+w]
            Answers.append(cropped_image)
        height, width, channels = imgr.shape
        sorted_good_contourss = []
        for con in good_contourss:
            sorted_good_contourss.insert(0,con)
        sorted_ans = []
        for ans in Answers:
            sorted_ans.insert(0,ans)
        loaded_file.append([sorted_ans,sorted_good_contourss,height,width])
    
    return loaded_file
def computer_vision_scanned_version(img , contours, height, width):
    img_scanned_resized = cv2.resize(img, (width, height))
    Answers = []
    for contour in contours:
        # getting the answer dimensions 
        x, y, w, h = cv2.boundingRect(contour)
        # cropping based on the dimensions
        cropped_image = img_scanned_resized[y:y+h, x:x+w]
        Answers.append(cropped_image)
    return Answers
def computer_vision_scanned_version_file_ide(img_files ,loaded_file):

    """
    This function take the loaded_file form computer_vision_soft_version_file_ide function and images of an scanned exam and return the sorted list 
    of answers

    img_files: scanned exam papers in order [form Pdf2image preferably]
    loaded_file: the file output ofcomputer_vision_soft_version_file_ide that is has been ran on the same img_files
    """
    answers = []
    for i in range(len(img_files)):
        img_scanned_resized =cv2.resize(img_files[i], (loaded_file[i][3], loaded_file[i][2]))
        answerss = []
        for contour in loaded_file[i][1]:
            # getting the answer dimensions 
            x, y, w, h = cv2.boundingRect(contour)
            # cropping based on the dimensions
            cropped_image = img_scanned_resized[y:y+h, x:x+w]
            answerss.append(cropped_image)
        answers.extend(answerss)
    return(answers)
