import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from src.models.emotionMapping import emotionID, emotionSentiment, classify_emotions

# Load tokenizer and trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("models/emotion")
model.eval()

# Set device (M1 Mac support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Prediction function
def predict_emotions(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)

    print(f"⚙️ Raw probabilities: {[round(p, 3) for p in probs[0].tolist()]}")

    predictedEmotions = (probs[0] > threshold).nonzero(as_tuple=True)[0].tolist()

    return predictedEmotions

# Map emotions to sentiment
def map_to_sentiment(emotions):
    sentiments = set()
    for emotion in emotions:
        sentiment = emotionSentiment.get(emotion, "neutral")
        sentiments.add(sentiment)

    if "positive" in sentiments and "negative" not in sentiments:
        return "positive"
    elif "negative" in sentiments and "positive" not in sentiments:
        return "negative"
    elif "positive" in sentiments and "negative" in sentiments:
        return "mixed"
    else:
        return "neutral"

# Main loop
if __name__ == "__main__":
    print("🧪 Emotion & Sentiment Model Tester (type 'exit' to quit)")
    while True:
        text = input("\nEnter a review: ")
        if text.lower() == "exit":
            break

        emotions = predict_emotions(text)
        sentiment = map_to_sentiment(emotions)
        
        # Map emotion IDs to names
        emotionNames = [emotionID.get(emotion, "Unknown") for emotion in emotions]

        print(f"\nPredicted Emotions: {emotionNames}")
        print(f"Mapped Sentiment: {sentiment}")
