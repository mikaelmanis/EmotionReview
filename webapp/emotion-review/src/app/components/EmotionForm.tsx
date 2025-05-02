'use client'

import { useState } from 'react'
import { predictEmotion } from '../utils/api';

export default function EmotionForm() {
  const [text, setText] = useState("");
  const [inputText, setInputText] = useState("");
  const [emotions, setEmotions] = useState<string[]>([]);
  const [sentiment, setSentiment] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    try {
      const result = await predictEmotion(text);
      setInputText(result.text);
      setEmotions(result.emotions);
      setSentiment(result.sentiment);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full max-w-[1200px]">
          <textarea
            className="w-full h-32 p-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your text here..."
          />
          <button
            type="submit"
            className="bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition duration-200"
          >
            {loading ? "Analyzing..." : "Predict emotions"}
          </button>
        </form>
        {emotions.length > 0 && (
        <div className="mt-6">
          <h2 className="text-2xl font-bold">For the given text</h2>
          <ul className="text-lg">{inputText}</ul>
          <h2 className="text-xl font-semibold">Predicted Emotions:</h2>
          <ul className="list-disc pl-6">
            {emotions.map((emo) => (
              <li key={emo}>{emo}</li>
            ))}
          </ul>
          <h2 className="text-xl font-semibold mt-4">Predicted Sentiment:</h2>
          <p className="text-lg">{sentiment}</p>
        </div>
      )}
    </div>
  )
}