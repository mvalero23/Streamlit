import streamlit as st
import numpy as np
import pickle

# Cargar el modelo y el vectorizador TF-IDF
with open('logistic_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

st.title("Predict Reviews' Sentiment")

# Función para limpiar el texto
def clean_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar signos de puntuación
    text = ''.join(char for char in text if char not in string.punctuation)
    # Eliminar espacios adicionales
    text = ' '.join(text.split())
    return text

# Entrada de la reseña
review = st.text_input("Ingrese la reseña:")

if st.button("Predicción"):
    # Limpiar y vectorizar la reseña
    review_cleaned = clean_text(review)
    review_vectorized = tfidf_vectorizer.transform([review_cleaned])

    # Predicción
    pred = modelo.predict(review_vectorized)[0]

    if pred == 0:
        st.write("La reseña es Negativa")
    else:
        st.write("La reseña es Positiva")