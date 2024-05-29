
import streamlit as st
import numpy as np
import pickle
from bs4 import BeautifulSoup
import re

# Cargar el modelo y el vectorizador TF-IDF
with open('logistic_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

st.title("Predict Reviews' Sentiment")

# Entrada de la reseña
review = st.text_input("Ingrese la reseña:")

if st.button("Predicción"):
    # Preprocesar y vectorizar la reseña
    review_cleaned = BeautifulSoup(review, "html.parser").get_text()
    review_cleaned = re.sub(r'[^\w\s]', '', review_cleaned)
    review_cleaned = review_cleaned.lower()
    review_vectorized = tfidf_vectorizer.transform([review_cleaned])

    # Predicción
    pred = modelo.predict(review_vectorized)[0]

    if pred == 0:
        st.write("La reseña es Negativa")
    else:
        st.write("La reseña es Positiva")
