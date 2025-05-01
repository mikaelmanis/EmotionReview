export async function predictEmotion(text: string) {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });
  
    if (!response.ok) {
      throw new Error("Failed to fetch prediction");
    }
  
    const data = await response.json();
    console.log("Response data:", data);
    return data;
  }