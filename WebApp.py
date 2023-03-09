import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps


vgg16 = load_model('vgg16.h5')

st.set_page_config(page_title='ASL Alphabet Recognition', layout='wide')
st.header("ASL Alphabet Recognition")
# st.footer("Project by Abhishek Biswas")
st.write("This is a Machine Learning Model Trained to recognise ASL Alphabets")

file = st.file_uploader("Kindly upload an image here", type = ['jpg', 'png'])

def pred(img):
    size = (224, 224,)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = np.array(img)
    img = img/255.
    img_reshape = img[np.newaxis, ...]
    p = vgg16.predict(img_reshape)

    return p

if file is not None:
    img = Image.open(file)
    st.image(img, width = 300 )
    p = pred(img)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
    st.success(s)