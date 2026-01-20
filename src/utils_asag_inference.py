import pandas as pd
import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch
#from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#import amrlib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import os
import joblib
import torch
import numpy as np
import pickle
import textdistance


def Gradanizer(num, grade_mode="fair"):
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
# Length Check
def length_check(student_answer, model_answer, threshold=0.2):
    """
    This function checks that the length of the student answer is sufficient to represent the fact mentioned in the model answer.
    If the student answer is too small then an automatic zero is assigned. Determining if it's too small should be a function of the length of the model answer.
    Returns boolean, True means the length is sufficient, False means it isn't which is an automatic 0.

    :param student_answer: str, the student's answer
    :param model_answer: str, the model answer
    :param threshold: float, the fraction of the model answer length that the student answer must meet or exceed
    :return: bool, True if the student answer length is sufficient, False otherwise
    """
    # Calculate the lengths of both answers
    student_length = len(student_answer.split())
    model_length = len(model_answer.split())
    # Calculate the minimum required length of the student answer
    required_length = model_length * threshold
    # Check if the student answer meets the required length
    return student_length >= required_length
def length_check_all(student_answer, model_answer, threshold=0.2):
    zero_flag = []
    for i, j in zip(student_answer, model_answer):
        zero_flag.append(length_check(i, j, threshold=0.2))
    return zero_flag

def load_model_tokenizer(model_path):
    model = AutoModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def text_tokenization(input_text, model, tokenizer, device="cpu"):
    batch_encoding = tokenizer.encode_plus(input_text, return_tensors="pt")
    tokenized_inputs = batch_encoding["input_ids"]
    tokenized_inputs = tokenized_inputs[:, :512].to(device)
    outputs = model(tokenized_inputs)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    return attention, tokenized_inputs, batch_encoding


def calculate_total_attention(attention):
    layer_sums = np.zeros((1, attention[0][0][0].shape[0]))
    for layer in attention:
        head_sums = np.zeros((1, layer[0][0].shape[1]))
        for head in layer[0].cpu():
            head = head.detach().numpy()
            head_sums += np.sum(head, axis=0)
        layer_sums += head_sums
    return layer_sums[0]


def filter_tokens(inputs, layer_sums, stop_words_tokens):
    ids = inputs[0].cpu().detach().numpy()
    out = [
        101,
        102,
        1010,
        1011,
        1012,
        100,
        1005,
        1025,
        1026,
        1027,
        1006,
        1007,
        1031,
        1032,
        1000,
    ]
    out.extend(stop_words_tokens)
    mask1 = np.ones(ids.shape, dtype=bool)
    for i in range(len(mask1)):
        if ids[i] in out:
            mask1[i] = 0
    ids = ids[mask1]
    layer_sums = layer_sums[mask1]
    return ids, layer_sums, mask1


def arbitrary_threshold(layer_sums, ids, threshold=70):
    # get 90 percentile of layer sums
    if layer_sums.size > 0:
        threshold = np.percentile(layer_sums, threshold)
    mask2 = np.zeros(layer_sums.shape, dtype=bool)
    for i, k in enumerate(layer_sums):
        if k >= threshold:
            mask2[i] = 1
    ids = ids[mask2]
    layer_sums = layer_sums[mask2]
    return ids, layer_sums, mask2


def get_word_indices(mask1, mask2):
    indices = np.arange(0, len(mask1))
    indices = indices[mask1]
    indices = indices[mask2]
    return indices


def get_corresponding_spans(batch_encoding, indices):
    all_spans = []
    for i in indices:
        lis = [batch_encoding.token_to_chars(i)[0], batch_encoding.token_to_chars(i)[1]]
        all_spans.append(lis)
    return all_spans


def spans_to_words(all_spans, input_text):
    words = []
    for i in all_spans:
        words.append(input_text[i[0] : i[1]])
    return words


def extract_attention_words(stop_words_tokens, model, tokenizer, input):
    attention, tokenized_inputs, batch_encoding = text_tokenization(
        input, model, tokenizer
    )
    layer_sums = calculate_total_attention(attention)
    ids, layer_sums, mask1 = filter_tokens(
        tokenized_inputs, layer_sums, stop_words_tokens
    )
    ids, layer_sums, mask2 = arbitrary_threshold(layer_sums, ids, 20)
    indices = get_word_indices(mask1, mask2)
    all_spans = get_corresponding_spans(batch_encoding, indices)
    words = spans_to_words(all_spans, input)
    return words

def strip_all(list_of_words):
    stripped = []
    for i in list_of_words:
        stripped.append(i.strip())
    return stripped

def overlap_association(student_answers, model_answers, model, tokenizer):
    stop_words_tokens= []
    overlap = []
    for model_answer, student_answer in tqdm(zip(student_answers, model_answers)):
        # extract the attention words from the model answer
        model_attention = strip_all(
            extract_attention_words(stop_words_tokens, model, tokenizer, model_answer)
        )
        answer_attention = strip_all(
            extract_attention_words(stop_words_tokens, model, tokenizer, student_answer)
        )
        overlap_student_model = len(
            set(model_attention).intersection(set(answer_attention))
        ) / len(set(model_attention).union(set(answer_attention)))
        overlap.append(overlap_student_model)
    return overlap

def text_similarity(left, right, ALG):
    actual = ALG(qval=1, algorithm=textdistance.jaro_winkler).similarity(left, right)
    return actual

def words_overlap(student_answers, model_answers, me = textdistance.MongeElkan()):
    monge_elkan_overlap = []
    for model_answer, student_answer in zip(student_answers, model_answers):
        monge_elkan_overlap.append(text_similarity(student_answer, model_answer, me))
    return monge_elkan_overlap 

def load_sentence_transformer():
    """
    This function loads the trained Sentence Transformer from Hugging Face.
    It returns two variables, the tokenizer and the model.
    """
    model_name = "youssefsameh1/ASAG-Fine-Tuned-ST-Model"
    model = SentenceTransformer(model_name)
    return model

def encode_wrong(wrong_answers, sent_model):
    """
    Use sentence transformers to encode the wrong answers into vectors, then pickle these vectors into a file
    which can then be used during deployment.

    Parameters:
    wrong_answers (list of str): List of wrong answer strings to be encoded.

    Returns:
    None: Just creates a pickle file in the current directory.
    """
    vectors = sent_model.encode(wrong_answers)
    with open('pickled_lists/encoded_wrong_answers.pkl', 'wb') as f:
        pickle.dump(vectors, f)

def compare_wrong_to_student(student_answer, sent_model, wrong_vectors, student_vector = None):
    """
    This code encodes the student answer then compares it (calculates cosine similarity) to the vectors inside the file generated by the encode_wrong function.
    If the similarity is bigger than the threshold, then the grade given should be 0.
    This function returns boolean, True if no significant similarity exists, False if it does.

    Parameters:
    student_answer (str): The answer given by the student.
    threshold (float): The cosine similarity threshold above which the student answer is considered significantly similar to a wrong answer.

    Returns:
    bool: True if no significant similarity exists, False if it does.
    """
    similarities = np.max(cosine_similarity(student_vector, wrong_vectors), axis=1)
    return similarities

def apply_compare_to_all(student_answers, sent_model, student_vector):
    with open('pickled_lists/wrong_answers.pkl', 'rb') as f:
        wrong_answers = pickle.load(f)
    if not os.path.exists("pickled_lists/encoded_wrong_answers.pkl"):
        encode_wrong(wrong_answers, sent_model)
    with open('pickled_lists/encoded_wrong_answers.pkl', 'rb') as f:
        wrong_vectors = pickle.load(f)
    all_compares = compare_wrong_to_student(student_answers, sent_model, wrong_vectors, student_vector)
    return all_compares


def norm_eval_sbert(embeddings_1=None, embeddings_2=None):
    y_pred = [float(util.cos_sim(e1, e2)[0][0]) for e1, e2 in zip(embeddings_1, embeddings_2)]
    return y_pred

def inference_sentence_transformer_all(student_answers, model_answers, final_model):
    students_embed=final_model.encode(student_answers)
    model_answer_embed=final_model.encode(model_answers)
    y_pred = norm_eval_sbert(students_embed, model_answer_embed)
    return y_pred, students_embed

def load_bert_model(path = "AmrSheta/asag_scibert_based"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer

#def load_paraphrasing_model(stog_path= "model_parse_xfm_bart_large-v0_1_0", gtos_path= "model_generate_t5wtense-v0_1_0"):
 #   stog = amrlib.load_stog_model(model_dir=stog_path)
 #   gtos = amrlib.load_gtos_model(model_dir=gtos_path)
 #   return stog, gtos

def paraphrase_text(student_answers, model_answers, stog, gtos):
    graphs_sa = stog.parse_sents(student_answers, disable_progress= False)
    sents_sa, _ = gtos.generate(graphs_sa, disable_progress= False)
    graphs_ma = stog.parse_sents(model_answers, disable_progress= False)
    sents_ma, _ = gtos.generate(graphs_ma, disable_progress= False)
    return sents_sa, sents_ma

def inference_bert(model, tokenizer, paraphrased_student, paraphrased_model, std=0.21825001144334838, mu=0.8246356696738376):
    model.eval()
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize the entire dataset at once
    inputs = tokenizer(
        paraphrased_student,
        paraphrased_model,
        max_length=256,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    ).to(device)

    # Model inference
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()

    # Move logits to CPU and scale outputs
    outputs = logits.to("cpu").numpy()
    scaled_outputs = (outputs * std) + mu

    return scaled_outputs

def initilize_models_asag():
    me = textdistance.MongeElkan
    sent_model = load_sentence_transformer()
    bert_model, bert_tokenizer = load_bert_model()
    paraphrase_model = None
    wrong_threshold = 0.9 # specific to the compare wrong fucntion in voiyu section
    sent_model = sent_model.to(torch.device("cpu"))
    bert_model = bert_model.to(torch.device("cpu"))
    keyword_model = AutoModel.from_pretrained("gpt2", output_attentions=True).to("cpu")
    keyword_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    random_forest_ensemble = joblib.load("models/random_forest.pkl")
    return me, sent_model, bert_model, bert_tokenizer, paraphrase_model, wrong_threshold, keyword_model, keyword_tokenizer, random_forest_ensemble

def inference_asag(student_answer, model_answer, sent_model, bert_model, bert_tokenizer, random_forest_model, monge_elkan, keyword_model, keyword_tokenizer, stog = None, gtos = None, paraphrase = False):
    if paraphrase:
        student_answer, model_answer = paraphrase_text(student_answer, model_answer, stog, gtos)
    zero_flag = length_check_all(student_answer, model_answer, threshold=0.3)
    sent_prediction, student_embeds = inference_sentence_transformer_all(student_answer, model_answer, sent_model)
    bert_prediction = inference_bert(bert_model, bert_tokenizer, student_answer, model_answer)
    monge_prediction = words_overlap(student_answer, model_answer, me = monge_elkan)
    keyword_overlap = overlap_association(student_answer, model_answer, keyword_model, keyword_tokenizer)
    records = [list(record) for record in zip(bert_prediction, sent_prediction, monge_prediction, keyword_overlap)]
    predictions = random_forest_model.predict(records)
    wrong_student = apply_compare_to_all(student_answer, sent_model, student_vector = student_embeds)
    zeroed_predictions = [0 if not zero or wrong*0.8>i else i for i, zero, wrong in zip(predictions, zero_flag, wrong_student)]
    final_predictions = [Gradanizer(i) for i in zeroed_predictions]
    return final_predictions    