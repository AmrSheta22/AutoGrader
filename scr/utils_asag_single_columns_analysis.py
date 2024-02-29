#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to classify question types
def classify_question_type(question):
    if re.search(r'\b(True|False)\b', question, re.IGNORECASE):
        return 'True/False'
    elif re.search(r'\b(Explain|Describe|Discuss|Definition|Define|Meaning|Explanation)\b', question, re.IGNORECASE):
        return 'Explanation'
    elif re.search(r'^(What|Where|When|Who|Why|How)\s', question, re.IGNORECASE):
        return 'WH-Start'
    elif re.search(r'\b\d+\s*[\+\-\*/]\s*\d+\b', question):
        return 'Math'
    elif re.search(r'^(Do|Does|Is|Are|Can|Will|Would|...)\s', question):
        return 'Verb Start'
    
    else:
        return 'Other'


# In[ ]:


# Function to perform Named Entity Recognition on a single text
def perform_ner_single(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# In[ ]:


# Function to calculate vocabulary
def calculate_vocabulary(text):
    words = text.lower().split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return len(set(words))


# In[ ]:


# Function to get POS Tags
def tokenize_and_pos_tag(text):
    words = text.lower().split()  # Simple split may be faster
    return pos_tag(words)


# In[ ]:


# Function that drops the null values rows from the column we want to work on
def remove_nulls_and_empty(column):
    # Drop rows with null values in the specified column
    column_without_nulls = column.dropna()

    # Remove rows with empty strings in the specified column
    column_without_empty = column_without_nulls[column_without_nulls != ' ']

    return column_without_empty


# In[ ]:




