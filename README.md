# EmotionReview

EmotionReview is an AI-powered web application that detects emotions and sentiment in text using a fine-tuned BERT-based model. It's designed to help users gain insights into the emotional tone of written content — from product reviews to social posts.

## Features

- Detects 28 distinct emotions from text input
- Maps emotions to overall sentiment (positive, negative, neutral, mixed)
- Interactive frontend built with Next.js and React
- Backend powered by FastAPI and Hugging Face Transformers
- Deployed on Vercel (frontend) and Render (backend)

## Tech Stack

- **Frontend:** Next.js, React, TypeScript
- **Backend:** Python, FastAPI, PyTorch
- **Model:** `bert-base-uncased` fine-tuned on GoEmotions
- **Hosting:** Vercel (web), Render (API)

## Installation to run locally

### Frontend

```bash
cd webapp/emotion-review
npm install
npm run dev
```

### Backend
```bash
cd render
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

## Live hosting

- [Frontend website](https://emotion-review.vercel.app)
- [Backend API](https://emotionmodelapi.onrender.com/)
- [Hugging Face model](https://huggingface.co/MikaelMani/emotion-model) (Custom model trained by me, uploaded for easier use)

Made By: Mikael Máni Eyfeld Clarke