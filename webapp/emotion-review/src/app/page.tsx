import Image from 'next/image'
import Link from 'next/link'
import EmotionForm from './components/EmotionForm';

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] justify-items-center min-h-screen sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[16px] row-start-2 items-center sm:items-start">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-[-.01em] text-center sm:text-left">
          Emotion Review
        </h1>
        <p className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
          This is a simple app that can predict your emotions on a text you provide using a trained prediction model.
        </p>
        <p className="text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
          The model was trained on movie reviews so it works best if the text is in the format of a movie review.
        </p>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-[-.01em] text-center sm:text-left">
          How to use it
        </h2>
        <ol className="list-decimal list-inside text-lg sm:text-xl text-center sm:text-left max-w-[600px] font-medium text-foreground/70">
          <li>Type your text in the box</li>
          <li>Click on the "Predict emotions" button</li>
          <li>See the predicted emotions!</li>
        </ol>
        <EmotionForm />
        <footer className="flex flex-row gap-13 sm:items-start text-sm sm:text-base text-foreground/70">
      <Link
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://github.com/mikaelmanis/EmotionReview"
          target="_blank"
          rel="noopener noreferrer"
        >
        <Image
            aria-hidden
            src="/github.svg"
            alt="Github icon"
            width={16}
            height={16}
          />
          GitHub
        </Link>
        <Link
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="/about"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/about.svg"
            alt="about icon"
            width={16}
            height={16}
          />
          About this project
        </Link>
        <a
          className="flex items-center gap-2"
          rel="noopener noreferrer"
          >
          <Image
            aria-hidden
            src="/user.svg"
            alt="user icon"
            width={16}
            height={16}
          />
          Mikael Máni Eyfeld Clarke
          </a>
      </footer>
      </main>
    </div>
  );
}
