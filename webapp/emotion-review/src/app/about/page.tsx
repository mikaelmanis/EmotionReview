import Link from "next/link";

export default function About() {
    return (
        <div className="grid grid-rows-[20px_1fr_20px] justify-items-center min-h-screen sm:p-20 font-[family-name:var(--font-geist-sans)]">
        <main className="flex flex-col gap-[16px] row-start-2 items-center sm:items-start">
            <h1 className="text-4xl sm:text-5xl font-bold tracking-[-.01em] text-center sm:text-left">
            About
            </h1>
            <p className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
            EmotionReview is an AI-powered tool that analyzes the emotional tone and sentiment behind movie reviews. However the model can also be used on a normal text such as from a review about a product or just a simple text that you write. The model was trained on movie reviews so it works best if the text is in the format of a movie review.
            </p>
            <h1 className="text-4xl sm:text-5xl font-bold tracking-[-.01em] text-center sm:text-left">
            What it does
            </h1>
            <p className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
            Using a fine-tuned BERT-based model, EmotionReview detects 28 distinct emotions — such as joy, sadness, anger, and excitement — and maps them to an overall sentiment: positive, negative, mixed, or neutral. The model has been trained on real-world data to recognize subtle emotional cues in human language.
            </p>
            <h1 className="text-4xl sm:text-5xl font-bold tracking-[-.01em] text-center sm:text-left">
            How it works
            </h1>
            <p className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
            The backend runs a custom-trained deep learning model that is uploaded on Hugging Face, hosted via a FastAPI server and integrated into this Next.js web application. When you input text, it's tokenized, processed by the model, and returned with both the predicted emotions and sentiment classification.
            </p>
            <Link href="https://huggingface.co/MikaelMani/emotion-model" className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
            Hugging Face Model
            </Link>

        </main>
        </div>
    );
    }