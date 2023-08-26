import streamlit as st
import pickle
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
df = pickle.load(open('dataset.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

st. markdown("<h1 style='text-align: center;'>Fashion Lens</h1>", unsafe_allow_html=True)

def extract_index_from_filename(filename):
    try:
        index = int(filename.split('_')[1].split('.')[0])
        return index
    except:
        return -1

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)
    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=13, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        cola, colb, colc = st.columns(3)

        with colb:
            st.image(display_image, use_column_width=True)
            st. markdown("<h5 style='text-align: center;'>Your Uploaded Image</h5>", unsafe_allow_html=True)

        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        
        # recommendention
        indices = recommend(features,feature_list)
        st. markdown("<h4 style='text-align: center;'>Some Similar Products are</h4>", unsafe_allow_html=True)
        num_rows = 3
        num_cols = 4
        total_images = num_rows * num_cols

        for i in range(num_rows):
            row = st.columns(num_cols)
            for j in range(num_cols):
                index = extract_index_from_filename(filenames[indices[0][i * num_cols + j]])
                if index != -1:
                    image_name = df['product_name'][index]
                    image_url = df['product_url'][index]
                    with row[j]:
                        st.image(filenames[indices[0][i * num_cols + j]])
                        st.write(f'<a href="{image_url}" target="_blank">{image_name}</a>', unsafe_allow_html=True)
    else:
        st.header("Some error occured in file upload")