from flask import Flask, render_template, request
import string
import warnings
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 

warnings.filterwarnings('ignore')


nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load chatbot data
with open("chatbot_copy.txt", 'r', errors='ignore') as f:
    raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Initialize Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    Tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = Tfidfvec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.pop(-1)

    if req_tfidf == 0:
        return "I am sorry! I don't understand you."
    else:
        return sent_tokens[idx]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get', methods=['POST'])
def chatbot_response():
    user_text = request.form['msg']
    return response(user_text)


if __name__ == '__main__':
    app.run(debug=False)
