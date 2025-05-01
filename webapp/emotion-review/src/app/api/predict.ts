import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const { text } = req.body

    const response = await axios.post('http://localhost:8000/predict', { text })

    res.status(200).json(response.data)
  } catch (error) {
    console.error(error)
    res.status(500).json({ error: 'Failed to get prediction' })
  }
}
