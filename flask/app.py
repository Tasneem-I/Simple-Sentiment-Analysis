from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

app = Flask(__name__)

# Loading to use DistilBERT model and tokenizer for sentiment analysis
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def get_sentiment_label(score):
    if score >= 0.65:
        return "Confident"
    elif 0.4 <= score < 0.65:
        return "Guessing"
    else:
        return "Uncertain"

def get_sentiment_type(label):
    if label == "LABEL_1":
        return "Positive"
    else:
        return "Negative"

#default home page route
@app.route('/')
def home():
    return render_template("/index.html")

#connection to make sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = classifier(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    score = get_sentiment_label(score)
    sentiment = get_sentiment_type(sentiment)
    return render_template('index.html', text=text, sentiment=sentiment, score=score)

if __name__ == '__main__':
    app.run(debug=True)
