import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Language Detection", page_icon="🌐")
st.subheader('🌐 Language Detection')

model_path = 'pages\model\language_model.pkl'
vectorizer_path = 'pages\model\language_vectorizer.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)
# Function to detect the language of a text
def detect_language(transformed_text):
    # Preprocess the input text if needed
    # preprocessed_text = transformed_text.lower()  # Example: converting to lowercase
    # vectorizer = CountVectorizer()
    # # Tokenize and transform the preprocessed text using the same vectorizer used during training
    # transformed_text = vectorizer.transform([preprocessed_text]).toarray()
    
    # Predict the language using the trained model
    predicted_language = model.predict(transformed_text)[0]
    
    return predicted_language


user = st.text_input("Enter something: ")


button = st.button('Detect')
if button:
    # cv = CountVectorizer()
    # vectorizer = CountVectorizer()
    # Tokenize and transform the preprocessed text using the same vectorizer used during training
    transformed_text = vectorizer.transform([user]).toarray()
    output = detect_language(transformed_text)
    st.markdown(f"This language is: {output}")
    print(output)

