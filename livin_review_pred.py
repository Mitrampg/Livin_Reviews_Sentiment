import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import joblib
import pickle
import contractions
import os
import string
import re
import nltk
import keras



# to make this notebook's output stable across runs
np.random.seed(42)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input, LSTM, GRU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TextVectorization, Embedding
from contractions import contractions_dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from googletrans import Translator
from langdetect import detect

import warnings
warnings.filterwarnings(action='ignore')

st.header('Milestone 2 Phase 2')
st.write("Nama: Mitra Marona Putra Gurusinga")
st.write("Batch: HCK 006")

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    text = " ".join([word for word in words if word.lower() not in stop_words])

    nltk.download('wordnet')
    lemmat = WordNetLemmatizer()
    text = " ".join(lemmat.lemmatize(word) for word in text.split())

    return text
def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='id', dest='en')
    return translated_text.text
def detect_language(text):
    return detect(text)

st.title('Model Prediction Livin Mandiri App Review Sentiment')
st.write('Please input your review')
st.write('NOTE = Only support Indonesian OR English')
st.write('NOTE = Please refresh if the Predict Button does not translate your review to english')
# Input data
teks = st.text_area('Review Text')

model = joblib.load("livin_app_review_pred.pkl")

if st.button('Predict'):
    # Translasi jika bahasa bukan Inggris
    detected_lang = detect_language(teks)
    if detected_lang == 'id':
        teks = translate_to_english(teks)
    else:
        teks = teks
    st.write('Your review:')
    st.write(teks)
    preprocessed_teks = preprocess_text(teks)
    # Prediksi dengan model
    prediction = model.predict([preprocessed_teks])
    prediction_labels = ["Good" if pred[0] > 0.5 else "Bad" for pred in prediction]

    st.write('This is categorized as: ')
    st.write(prediction_labels)