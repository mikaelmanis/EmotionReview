from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from app.emotionMapping import emotionID, emotionSentiment

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://emotion-review.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Loads the model I trained from the Huggingface hub
model = BertForSequenceClassification.from_pretrained("MikaelMani/emotion-model").to(device)
model.eval()

# Creates a class to define the input data structure
# This is used to validate the input data
class InputText(BaseModel):
    text: str

# Post request to predict sentiment
# This endpoint receives a text input and returns the predicted emotions and sentiment, allowing for the frontend to display the results
@app.post("/predict")
async def predict_sentiment(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0]

    predicted_indices = (probs > 0.1).nonzero(as_tuple=True)[0].tolist()
    emotions = [emotionID[i] for i in predicted_indices]

    sentiments = set()
    for emotion in predicted_indices:
        sentiment = emotionSentiment.get(emotion, "neutral")
        sentiments.add(sentiment)

    if "positive" in sentiments and "negative" not in sentiments:
        sentiment = "positive"
    elif "negative" in sentiments and "positive" not in sentiments:
        sentiment = "negative"
    elif "positive" in sentiments and "negative" in sentiments:
        sentiment =  "mixed"
    else:
        sentiment = "neutral"

    return {
        "text": input.text,
        "emotions": emotions,
        "sentiment": sentiment
    }
