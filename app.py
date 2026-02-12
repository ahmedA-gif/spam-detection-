import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import os
nltk.download('punkt')
nltk.download('stopwords')

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', force=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', force=True)

# Load model and vectorizer
try:
    model = joblib.load('spam_classifier.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please run spam_classifier.py first to generate them.")
    st.stop()
for resource in ['punkt_tab', 'punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}') if resource != 'stopwords' else nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resourc

# Load dataset for similar messages
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
except:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    df = pd.read_csv(url, compression='zip', sep='\t', names=['label', 'message'], encoding='latin-1')
df = df[['label', 'message']].dropna()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess dataset text for similar message lookup
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
df['processed_text'] = df['message'].apply(
    lambda text: ' '.join([ps.stem(word) for word in word_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower())) if word not in stop_words])
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .header { 
        background-color: #007bff; 
        padding: 20px; 
        color: white; 
        text-align: center; 
        border-radius: 10px; 
        margin-bottom: 20px; 
    }
    .header-title { font-size: 28px; font-weight: bold; margin: 0; }
    .nav-bar { 
        background-color: #343a40; 
        padding: 10px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
    }
    .nav-bar a { 
        color: white; 
        text-decoration: none; 
        margin: 0 20px; 
        font-size: 18px; 
    }
    .nav-bar a:hover { color: #f8f9fa; text-decoration: underline; }
    .nav-bar a.active { font-weight: bold; text-decoration: underline; }
    .stTextArea textarea { 
        border-radius: 10px; 
        padding: 15px; 
        font-size: 16px; 
        border: 2px solid #ced4da; 
    }
    .stButton button { 
        background-color: #007bff; 
        color: white; 
        border-radius: 10px; 
        padding: 10px 20px; 
        font-size: 16px; 
        border: none; 
    }
    .stButton button:hover { background-color: #0056b3; }
    .result-box { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
        margin-top: 20px; 
    }
    .keyword-chip { 
        background-color: #e9ecef; 
        padding: 5px 10px; 
        border-radius: 15px; 
        margin: 5px; 
        display: inline-block; 
        font-size: 14px; 
    }
    .footer { 
        background-color: #343a40; 
        color: white; 
        text-align: center; 
        padding: 15px; 
        border-radius: 10px; 
        margin-top: 20px; 
        font-size: 14px; 
    }
    .subheader { font-size: 20px; color: #343a40; font-weight: bold; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><span class="header-title">SMS Spam Classifier</span></div>', unsafe_allow_html=True)

# Navigation bar
nav_selection = st.sidebar.selectbox("Navigate", ["Home", "About", "Visualizations"])

# Main content
if nav_selection == "Home":
    st.markdown("Enter an SMS message to classify it as **SPAM** or **HAM**. Get detailed insights including spam probability and key words.", unsafe_allow_html=True)

    # Input form
    with st.form(key='message_form'):
        message = st.text_area("Enter your message:", placeholder="Type your message here...", height=100)
        submit_button = st.form_submit_button(label="Classify Message")

    # Process and predict
    if submit_button and message:
        text = message.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        try:
            words = word_tokenize(text)
        except LookupError:
            nltk.download('punkt', force=True)
            words = word_tokenize(text)
        words = [ps.stem(word) for word in words if word not in stop_words]
        processed_text = ' '.join(words)
        vector = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]
        feature_names = tfidf.get_feature_names_out()
        feature_values = vector[0]
        important_words = [word for word, val in zip(feature_names, feature_values) if val > 0]

        # Display results
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f'**Message**: {message}')
        st.markdown(f'**Prediction**: {"SPAM" if prediction == 1 else "HAM"}')
        st.markdown(f'**Spam Probability**: {probability:.2%}')
        
        # Display key words
        st.markdown('<div class="subheader">Key Words Influencing Prediction:</div>', unsafe_allow_html=True)
        if important_words:
            for word in important_words:
                st.markdown(f'<span class="keyword-chip">{word}</span>', unsafe_allow_html=True)
        else:
            st.markdown("No significant keywords found.")

        # Display similar spam messages
        if prediction == 1 and important_words:
            similar_spam = df[(df['label'] == 1) & (df['processed_text'].str.contains('|'.join(important_words), regex=True))].sample(2, random_state=42) if important_words else pd.DataFrame()
            if not similar_spam.empty:
                st.markdown('<div class="subheader">Similar Spam Messages from Dataset:</div>', unsafe_allow_html=True)
                for idx, row in similar_spam.iterrows():
                    st.markdown(f"- {row['message']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif nav_selection == "About":
    st.markdown('<div class="subheader">About the SMS Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown("""
        This application uses a machine learning model to classify SMS messages as **SPAM** or **HAM** (non-spam). 
        The model is trained on the UCI SMS Spam Collection Dataset using a Voting Classifier that combines 
        Multinomial Naive Bayes and Logistic Regression, achieving **97-98% accuracy**. Key features include:

        - **Text Preprocessing**: Lowercasing, removing special characters, stemming, and stopword removal.
        - **Feature Extraction**: TF-IDF with bigrams for robust text representation.
        - **Model**: Ensemble of Naive Bayes and Logistic Regression with hyperparameter tuning.
        - **Insights**: Displays spam probability, influential keywords, and similar spam messages.
        - **Visualizations**: Includes confusion matrix, class distribution, top spam words, and word cloud.

        Run `spam_classifier.py` to train the model and generate visualizations. The Flask backend (`app.py`) provides API access.
    """)

elif nav_selection == "Visualizations":
    st.markdown('<div class="subheader">Model Visualizations</div>', unsafe_allow_html=True)
    if os.path.exists('spam_analysis.png'):
        st.image('spam_analysis.png', caption='Model Performance and Word Analysis (Confusion Matrix, Class Distribution, Top Words, Word Cloud)', use_column_width=True)
    else:
        st.warning("Visualization file (spam_analysis.png) not found. Please run spam_classifier.py to generate it.")

# Footer
st.markdown("""
    <div class="footer">
        Developed by AHMEDAI | Powered by Streamlit | © 2025
    </div>
""", unsafe_allow_html=True)
