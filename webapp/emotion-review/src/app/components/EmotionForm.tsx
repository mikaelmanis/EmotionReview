'use client'

import { useState } from 'react'

export default function EmotionForm() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })

    const data = await response.json()
    setResult(data)
  }

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
            Predict
          </button>
        </form>
        {result && (
        <pre className="w-full max-w-[600px] p-4 border border-gray-300 rounded-lg bg-gray-50 text-sm sm:text-base font-mono whitespace-pre-wrap overflow-x-auto">
        <h2 className="text-lg sm:text-xl font-bold mb-2">Predicted Emotions:</h2>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}