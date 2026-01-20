import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import math
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

def pridector(model, student_answer, model_answer, mode):
    """
    Pridector function take granding model and model and student answer and output a final grade
    
    Parameters 
    model : The grading model note most be Sentance transformers model 
    student_answer : the text of the student answer
    model_answer : the text of the model answer
    mode : the mode of gradanizer
    
    """
    
    def Gradanizer(num, grade_mode="Fair"):
        """
        This function take the garde form 0 to 1 and output that garde in form 0 to 5
        parameters
        num: the garde in form 0 to 1
        grade_mode: how fair you want to model to be [fair: the exact transformation with out any lose ranges, easy: more skewed into higher grades more 
        common, lose_ends: more skewed into higher grades and lower grades]
        """
        def get_region_value_f(number):    
            intervals = [
            (-10, 0.4545), (0.4545, 0.9091), (0.9091, 1.3636),
            (1.3636, 1.8182), (1.8182, 2.2727), (2.2727, 2.7273),
            (2.7273, 3.1818), (3.1818, 3.6364), (3.6364, 4.0909),
            (4.0909, 4.5455), (4.5455, 6)]
            values=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
            for i, (start, end) in enumerate(intervals):
                if start <= number < end:
                    return values[i]
        
        def get_region_value_e(number):    
            intervals = [
            (-10, 0.4545), (0.4545, 0.9091), (0.9091, 1.3636),
            (1.3636, 1.8182), (1.8182, 2.2727), (2.2727, 2.7273),
            (2.7273, 3.1818), (3.1818, 3.6364)]
            values=[0,0.5,1,1.5,2,2.5,3,3.5]
            for i, (start, end) in enumerate(intervals):
                if start <= number < end:
                    return values[i]
        def get_region_value_l(number):    
            intervals = [
            (1.0000, 1.3571), (1.3571, 1.7143), (1.7143, 2.0714),
            (2.0714, 2.4286), (2.4286, 2.7857), (2.7857, 3.1429),
            (3.1429, 3.5000)]
            values=[0.5,1,1.5,2,2.5,3,3.5]
            for i, (start, end) in enumerate(intervals):
                if start <= number < end:
                    return values[i]

        if (grade_mode == "fair"):
            num_f=num*5
            return get_region_value_f(num_f)
        elif (grade_mode == "easy"):
            if (num >=0.85):
                return 5
            elif ( 0.85>num>=0.8):
                return 4.5
            elif ( 0.8>num>=0.7):
                return 4
            else:
                num_e = num*5
                return get_region_value_e(num_e)
        elif (grade_mode == "lose_ends"):
            if (num >=0.85):
                return 5
            elif ( 0.85>num>=0.8):
                return 4.5
            elif ( 0.8>num>=0.7):
                return 4
            elif (0.2>=num):
                return 0
            else:
                num_l = num*5
                return get_region_value_l(num_l)
        else:
            print("not a valid mode ")
    
    sa= model.encode(student_answer,convert_to_tensor=True)
    ma= model.encode(model_answer,convert_to_tensor=True)
    grade = Gradanizer(util.cos_sim(sa, ma),mode)
    return grade
    