#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Gradanizer Function
def Gradanizer(num, grade_mode="Fair"):
    """
    This function take the grade form 0 to 1 and output that grade in form 0 to 5

    parameters:
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


# In[ ]:


# Evaluation Function
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
        return MAE,correlation_coefficient,r_squared
        #if you want the hit precentage tag the above and untag the blew
        #return MAE,correlation_coefficient,hits_precentage


# # First: BERT for Question Answering

# In[ ]:


# Function to calculate Embeddings 
def calculate_embeddings(texts):
    model.eval()
    embeddings = []

    for text in texts:
        # Tokenize and encode the input text
        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

        # Ensure the model returns hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract embeddings from a specific layer (adjust layer_index as needed)
        layer_index = 11  # You may need to adjust this based on your model architecture
        hidden_states = outputs.hidden_states[layer_index]

        # Use the [CLS] token representation (index 0) as the embedding
        embeddings.append(hidden_states[:, 0, :].cpu().numpy())

    return embeddings


# # Second: BERT by Sentence Transformers

# In[ ]:


# Function to load data
def data_loader_local(data=None):
    input_data = []
    for index, row in data.iterrows():
        student_answer = row['student_answer']
        model_answer = row['model_answer']
        grade = row['grade']
    
        # Create InputExample instances
        input_data.append(InputExample(texts=[student_answer, model_answer], label=grade))
    return(input_data)


# In[ ]:


# Function to Define the model
Model_bert = SentenceTransformer('bert-base-nli-mean-tokens')


# In[ ]:


# Function to Define a data loader
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)


# In[ ]:


# Function to Define a loss function (e.g., cosine similarity loss)
loss_function = losses.CosineSimilarityLoss(model=Model_bert)


# In[ ]:


# Fine-tune the model
Model_bert.fit(
    train_objectives=[(train_dataloader, loss_function)],
    epochs=5,
    warmup_steps=200,
    show_progress_bar=True
)


# In[ ]:


# Get student answer embeddings
student_embeddings=Model_bert.encode(test_data["student_answer"].to_list(),convert_to_tensor=True)


# In[ ]:


# Get model answer embeddings
model_embeddings=Model_bert.encode(test_data["model_answer"].to_list(), convert_to_tensor=True)

