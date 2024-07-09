
import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import math
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
def Modeling(val_data,train_data1,training_data2,mode = "easy",model_name = 'all-distilroberta-v1'):
    """
    This function is meant to preform modeling 
    val_data: a data frame that meant to be the the test data for our model
    train_data1 : a data frame that meant to be first training data set for our model
    train_data2 : a data frame that meant to be first training data set for our model
    mode : the grade_mode for our Gradanizer function  {grade_mode: how fair you want to model to be [fair: the exact transformation with out 
    any lose ranges, easy: more skewed into higher grades more common, lose_ends: more skewed into higher grades and lower grades]} 
    model_name : is the model name form sentance transfomers that you want the modeling to be about default is 'all-distilroberta-v1' because
    its the best currently for our model
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
            
            
    def Evaluation(embeddings_1=None, embeddings_2=None,test_data=None,mode="easy"):
        """
        this function takes two lists of embeddings in the form of torch tensor or list or df column one for the student answer and other for model answer
        or any two sentences if this function used in any other code and the test data frame and the mode of Gradazier and output MAE % and r_squared and 
        correlation_coefficient
        parameters
        embeddings_1:list or tensor or df column that hold the sentence embeddings (student answer embeddings)
        embeddings_2:list or tensor or df column that hold the sentence embeddings (model answer embeddings)
        data: the test data frame
        mode: the mode of which gradazier is used ["fair", "easy", "lose_ends"]
        """
        from sentence_transformers import util
        if (len(embeddings_1) != len(embeddings_2)):
            print("embeddings_1 and embeddings_2 are not the same length")
            return None
        elif (mode not in ["easy","lose_ends","fair"]):
            print("not valid mode")
            return None
        else:
            predicted = []
            for i in range(len(embeddings_1)):
                predicted.append(Gradanizer(util.cos_sim(embeddings_1[i], embeddings_2[i]),mode))
            transformed_grade=[]
            for i in range(len(embeddings_1)):
                #transformed_grade.append(Gradanizer(test_data["grade"][i],mode))
                transformed_grade.append(test_data["grade"][i]*5)
            arr_predicted =np.array(predicted)
            arr_grade=np.array(transformed_grade)
            MAE = (1-((np.sum(np.abs(arr_predicted-arr_grade))/len(embeddings_1))/5))*100
            correlation_coefficient, p_value = spearmanr(arr_grade, arr_predicted)
            r_squared = r2_score(arr_grade, arr_predicted)
            dif =arr_predicted-arr_grade
            hits_precentage= (np.count_nonzero(dif == 0)/len(embeddings_1))
            #return MAE,correlation_coefficient,r_squared
            #if you want the hit precentage tag the above and untag the blew
            return MAE,correlation_coefficient,hits_precentage
        
    def data_loader_local(data=None):
        input_data = []
        for index, row in data.iterrows():
            student_answer = row['student_answer']
            model_answer = row['model_answer']
            grade = row['grade']
        
            # Create InputExample instances
            input_data.append(InputExample(texts=[student_answer, model_answer], label=grade))
        return(input_data)
        
    final_model = SentenceTransformer(model_name)
    
    train_dataset_og = data_loader_local(train_data1)
    train_dataloader_og = DataLoader(train_dataset_og, shuffle=True, batch_size=16)
    train_loss_og = losses.CosineSimilarityLoss(model=final_model)
    final_model.fit(train_objectives=[(train_dataloader_og, train_loss_og)], epochs=5, warmup_steps=330)
    train_dataset_asag = data_loader_local(training_data2)
    train_dataloader_asag = DataLoader(train_dataset_asag, shuffle=True, batch_size=16)
    train_loss_asag = losses.CosineSimilarityLoss(model=final_model)
    final_model.fit(train_objectives=[(train_dataloader_asag, train_loss_asag)], epochs=1, warmup_steps=27)
    students_embed_asag=final_model.encode(val_data["student_answer"].to_list(),convert_to_tensor=True)
    model_answer_embed_asag=final_model.encode(val_data["model_answer"].to_list(),convert_to_tensor=True)
    precentage_2,correlation_coefficient_2,hit_precentage_2 = Evaluation(students_embed_asag,model_answer_embed_asag,val_data,mode)
    return precentage_2,correlation_coefficient_2,hit_precentage_2,final_model
