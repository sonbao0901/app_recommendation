import streamlit as st
import os
import pickle
from st_keyup import st_keyup
import tensorflow as tf
import numpy as np
import re
from utils import load_model_trained, load_tokenizer, get_prediction_eos, correct_word

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


st.set_page_config(layout = 'wide')


def given_text(text):
    if len(text.split()) == 1:
        res, res_top5 = get_prediction_eos(model, tokenizer, text)
        return [res, res_top5]
    else:
        text_char = text
        text_before = ' '.join(text.split()[:-1]) 
        pred_text_char, pred_text_char_top5 = get_prediction_eos(model, tokenizer, text_char)
        pred_text_before, pred_text_before_top5 = get_prediction_eos(model, tokenizer, text_before)
        if pred_text_char == pred_text_before:        
            answer = [text_before, pred_text_char]
            answer_as_string = " ".join(answer)
            return [answer_as_string, pred_text_char_top5]
        else:
            answer = [text_char, pred_text_char]
            answer_as_string = " ".join(answer)
            return [answer_as_string, pred_text_char_top5]


st.title("Next Word Prediction with Son Bao")

st.info('KEEP ENTERING A CHARACTER THEN IT WILL AUTOMATICALLY GENERATE THE PREDICTION. (WILL TAKE A BIT TIME DUE TO NOT DOING OPTIMIZATION.)')

input_text = st_keyup('Enter your text here:')

@st.cache_resource()
def load_model():
    return load_model_trained()
model  = load_model()

@st.cache_resource()
def load_token():
    return load_tokenizer()
tokenizer = load_token()

sp = True
col1, col2 = st.columns(2)
if sp:
    with col1:
        if correct_word(input_text)[1]:
            text = correct_word(input_text)
            res, top5 = correct_word(input_text)[0], correct_word(input_text)[2]
        else:
            text = input_text
            text = input_text.lower()
            text = re.sub(r'[^\w\s]+', ' ', text)
            res, top5 = given_text(text)
        st.text_area("Word Predict with Given text", res, key="Predicted_word")
        st.text_area("Predicted List is Here", top5, key="Predicted_list")    
    with col2:
        st.markdown("![Alt Text](https://media.tenor.com/zXhK-0R9y1gAAAAi/vengeful-notes.gif)")