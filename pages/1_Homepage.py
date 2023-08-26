import streamlit as st
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

st. markdown("<h3>Products You May Like</h3>", unsafe_allow_html=True)

file_path = os.path.join(os.path.dirname(__file__), "..", "dataset.pkl")
with open(file_path, "rb") as f:
    df = pickle.load(f)

file_path_1 = os.path.join(os.path.dirname(__file__), "..", "user_data.pkl")
with open(file_path_1, "rb") as f1:
    user_df = pickle.load(f1)

user_df.drop(0, inplace=True)
user_df.drop(2, inplace=True)
user_df = user_df.reset_index()
user_df.drop(['index'], axis='columns', inplace=True)


product_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
product_tfidf = product_vectorizer.fit_transform(df['cleaned_desc'])

user_tfidf = product_vectorizer.transform(user_df['cleaned_desc'])

cosine_similarities = np.zeros((user_tfidf.shape[0], product_tfidf.shape[0]))

# Calculating cosine similarities between user and all products
for i in range(user_tfidf.shape[0]):
    user_vector = user_tfidf[i]
    similarities = cosine_similarity(user_vector, product_tfidf)
    cosine_similarities[i] = similarities

for user_index, user_row in user_df.iterrows():
    # Calculate cosine similarities for the user
    user_vector = user_tfidf[user_index]
    cosine_similarities = cosine_similarity(user_vector, product_tfidf)

    most_similar_indices = np.argsort(cosine_similarities[0])[-2:][::-1]

    similar_product_names = [df.iloc[index]['product_name'] for index in most_similar_indices]
    similar_product_links = [df.iloc[index]['product_url'] for index in most_similar_indices]
    similar_image_links = [df.iloc[index]['image_url'] for index in most_similar_indices]

    product_names_list = []
    product_links_list = []
    image_links_list = []

    num_containers = 5
    num_columns = 3
    total_images = num_containers * num_columns

    containers = []

    for container_index in range(num_containers):
        images_in_container = []

        for col_index in range(num_columns):
            image_index = container_index * num_columns + col_index
            if image_index < len(similar_product_names):
                product_name = similar_product_names[image_index]
                product_link = similar_product_links[image_index]
                img_url = similar_image_links[image_index]

                img_style = f"width: 150px; height: 150px; object-fit: contain;"
                img_and_name_html = f'<img src="{img_url}" style="{img_style}" /><br><a href="{product_link}    " target="_blank">{product_name}</a>'

                images_in_container.append(img_and_name_html)

        if images_in_container:
            container_html = '<div style="display: flex;">'
            container_html += ''.join([f'<div style="flex: 1; padding: 5px;">{img_html}</div>' for img_html     in images_in_container])
            container_html += '</div>'
            containers.append(container_html)

    for container_html in containers:
        st.write(container_html, unsafe_allow_html=True)