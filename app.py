from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==================== NLTK Downloads ====================
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

app = Flask(__name__)

# ==================== OLD MODEL (LSTM) ====================
model = load_model('model.h5')
#tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
with open('tokenizer.pkl', 'rb') as f:
    content = f.read().replace(b'\r', b'')

tokenizer = pickle.loads(content)

#encoder = pickle.load(open('encoder.pkl', 'rb'))
with open('encoder.pkl', 'rb') as f:
    content = f.read().replace(b'\r', b'')

encoder = pickle.loads(content)

w2v_model = gensim.models.Word2Vec.load('model.w2v')

# Constants
SEQUENCE_LENGTH = 300
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# ==================== TEXT PREPROCESSING ====================
stop_words = set(stopwords.words("english"))
stop_words.remove('not')
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stop_words = stop_words.union(more_stopwords)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    html = re.compile(r'^[^ ]<.*?>|&([a-z0-9]+|#[0-9]\"\'\"{1,6}|#x[0-9a-f]{1,6});[^A-Za-z0-9]+')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_quotes(text):
    quotes = re.compile(r'[^A-Za-z0-9\s]+')
    return re.sub(quotes, '', text)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Apply preprocessing steps
    text = remove_URL(text)
    text = remove_emoji(text)
    text = remove_html(text)
    text = remove_punct(text)
    text = remove_quotes(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lower case and remove stopwords
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    
    # POS tagging and lemmatization
    pos_tags = nltk.tag.pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in pos_tags]
    
    return ' '.join(lemmatized)

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict_sentiment(text):

    print("Old model predicting...: ", text)

    # Preprocess text
    clean_text = preprocess_text(text)
    
    # Tokenize and pad
    x_test = pad_sequences(tokenizer.texts_to_sequences([clean_text]), maxlen=SEQUENCE_LENGTH)
    
    # Predict
    score = model.predict([x_test])[0][0]
    
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=True)
    
    return {
        "label": label, 
        "score": float(score),
        "confidence": max(score, 1-score) * 100
    }

# ==================== TRANSFORMER (V2) ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_name = "./best_model/distilbert-sentiment-model"
trans_tokenizer = AutoTokenizer.from_pretrained(transformer_name)
trans_model = AutoModelForSequenceClassification.from_pretrained(transformer_name).to(device)
trans_model.eval()

id2label = {1: NEGATIVE, 0: POSITIVE}

def predict_transformer(texts, top_k=1):

    print("V2 - Transformer model predicting...: ", texts)

    enc = trans_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        out = trans_model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    preds = np.argmax(probs, axis=-1)
    results = []
    for i, p in enumerate(preds):
        score = probs[i][p]
        results.append({
            "label": id2label[p],
            "score": float(score),
            "confidence": float(score * 100)
        })
    return results

# ==================== ROUTES ====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text'})
        
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# --- NEW V2 ROUTES ---
@app.route('/v2')
def home_v2():
    return render_template('index_v2.html')

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    try:
        data = request.get_json()
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Please enter some text'})
        result = predict_transformer([text])[0]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# ==================== MAIN ====================
if __name__ == '__main__':
    # app.run(debug=True)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
