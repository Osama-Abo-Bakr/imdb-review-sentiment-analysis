from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import pickle


model = load_model(r'D:\Pycharm\Computer Vision Project\NLP Project\IMDB Review\model.h5')
tokenizer = pickle.load(open(r'D:\Pycharm\Computer Vision Project\NLP Project\IMDB Review\tokenizer.bin', 'rb'))
class_label = {1: 'Good', 0: 'Bad'}
st.title('IMDB Review. 🎞️🎞️')

text = st.text_input('Enter THe Text')
if text:
    text = tokenizer.texts_to_sequences([text])
    pad_seq = np.array(pad_sequences(text, maxlen=936, padding="pre"))
    if st.button('predict'):
        prediction = model.predict(pad_seq)
        st.write(class_label[prediction.argmax()])
