import openai
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ast
import os
import speech_recognition as sr

file_path = os.path.join(os.path.dirname(__file__), "..", "dataset.pkl")
with open(file_path, "rb") as f:
    df = pickle.load(f)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def retrieve_similar_products(input_tokens, df):
    tfidf_v = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    tfidf_ma = tfidf_v.fit_transform(df['cleaned_desc'])

    input_tfi = tfidf_v.transform([' '.join(input_tokens)])

    cosine_similaritie = linear_kernel(input_tfi, tfidf_ma).flatten()
    cosine_threshold = 0.1
    
    similar_indices = [i for i, score in enumerate(cosine_similaritie) if score > cosine_threshold]

    stored_image_urls = []
    stored_product_names = []
    stored_product_urls = []

    for i in similar_indices:
        similar_images = df['image'][i]
        product_name = df['product_name'][i]
        product_url = df['product_url'][i]
        stored_product_names.append(product_name)
        stored_product_urls.append(product_url)
        try:
            image_list = ast.literal_eval(similar_images)
            if isinstance(image_list, list) and len(image_list) > 0:
                stored_image_urls.append(image_list[0])
        except (ValueError, SyntaxError):
            pass
    return stored_image_urls, stored_product_names, stored_product_urls

def display_images_with_names_and_links(image_urls, product_names, product_urls, container_index):
    num_images = len(image_urls)
    col_width = st.columns(4)

    for i in range(container_index, num_images, 3):

        with col_width[i % 4]:
            img_url = image_urls[i]
            product_name = product_names[i]
            product_url = product_urls[i]

            img_style = f"width: 150px; height: 150px; object-fit: contain;"

            img_and_name_html = f'<img src="{img_url}" style="{img_style}" /><br><a href="{product_url}" target="_blank">{product_name}</a>'

            st.write(f'<div style="text-align: center;">{img_and_name_html}</div>', unsafe_allow_html=True)

def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak something...")
        audio = r.listen(source)
    try:
        prompt = r.recognize_google(audio)
        return prompt
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    
def process_user_input(prompt):
    message_placeholder = st.empty()
    full_response = ""

    completion = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=1000)
    response_text = completion.choices[0]['text']

    full_response += "Here are some of the outfits -"
    message_placeholder.markdown(full_response + "▌")

    input_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokenizer.tokenize(response_text.lower()) if token.lower() not in stop_words]
    image_url, product_name, product_url = retrieve_similar_products(input_tokens, df)

    for container_index in range(3):
        with st.container():
            display_images_with_names_and_links(image_url, product_name, product_url, container_index)

    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

if not hasattr(st.session_state, "messages"):
    st.session_state.messages = []

st. markdown("<h1 style='text-align: center;'>Fashion Bot</h1>", unsafe_allow_html=True)

with st.sidebar:
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 51):
        st.warning('Please enter correct credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        process_user_input(prompt)

voice_input_placeholder = st.empty()

st.sidebar.write('Give voice input to Fashion Bot')

voice_search_button_text = "Voice Search🎤"
if st.sidebar.button(voice_search_button_text, key="voice_search", help="Voice Search"):
    with voice_input_placeholder:
        st.write("Listening...")
        voice_prompt = get_voice_input()

    if voice_prompt:
        st.session_state.messages.append({"role": "user", "content": voice_prompt})
        with st.chat_message("user"):
            st.markdown(voice_prompt)
        with st.chat_message("assistant"):
            process_user_input(voice_prompt)