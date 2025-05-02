# Map GOEmotion IDs to emotions
emotionID = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

emotionSentiment = {
    0: "positive",
    1: "positive",
    2: "negative",
    3: "negative",
    4: "positive",
    5: "positive",
    6: "neutral",
    7: "positive",
    8: "positive",
    9: "negative",
    10: "negative",
    11: "negative",
    12: "negative",
    13: "positive",
    14: "negative",
    15: "positive",
    16: "negative",
    17: "positive",
    18: "positive",
    19: "negative",
    20: "positive",
    21: "positive",
    22: "neutral",
    23: "positive",
    24: "negative",
    25: "negative",
    26: "positive",
    27: "neutral"
}

def classify_emotions(predicted_emotion_ids):
    sentiment_score = sum(
        1 if emotionSentiment.get(i) == "positive"
        else -1 if emotionSentiment.get(i) == "negative"
        else 0
        for i in predicted_emotion_ids
    )

    if sentiment_score > 0:
        return "positive"
    elif sentiment_score < 0:
        return "negative"
    else:
        return "neutral"
