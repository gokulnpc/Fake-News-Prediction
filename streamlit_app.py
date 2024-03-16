import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('stopwords')

# Function to load the model
@st.cache_data
def load_model():
    with open('fake_news_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

@st.cache_data
def load_vectorizer():
    with open('vectorizer', 'rb') as file:
        load_vectorizer = joblib.load(file)
    return load_vectorizer

# Load your model
loaded_model = load_model()

# Load your vectorizer
vectorizer = load_vectorizer()

def stemmer(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Function to create the input datafram
def create_input_df(user_inputs):
    user_inputs['content'] = user_inputs['content'].apply(stemmer)
    input_df = pd.DataFrame(user_inputs, index=[0])
    x = input_df['content'].values
    x = vectorizer.transform(x).toarray()
    return x
    

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Fake News Prediction Web App')

    # Author
    author = st.text_input('Author', 'Author name')
    # Title
    title = st.text_input('Title', 'Title of the article')
    # Text
    text = st.text_area('Text', 'Text of the article')

    user_inputs = { 
        'author': author,
        'title': title,
        'text': text
    }
    data = {}
    content = author + ' ' + title
    data['content'] = content
    data = pd.DataFrame(data, index=[0])

    if st.button('Predict'):
        input_df = create_input_df(data)
        prediction = loaded_model.predict(input_df)
        if prediction == 0:
            st.success('The news is real!')
        else:
            st.error('The news is fake!')
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Logistic Regression')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'fake_news_prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="fake_news_prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'train.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="fake_news.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Fake-News-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is a fake news prediction tool. It uses a machine learning model to predict whether a given news is fake or not. The model is trained on a dataset of news articles and their labels. The model uses a logistic regression algorithm to make predictions.')
    st.write('--'*50)

    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Fake-News-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
