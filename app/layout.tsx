import type { Metadata } from 'next';
import { Press_Start_2P, VT323, Space_Grotesk } from 'next/font/google';
import './globals.css';

const pressStart = Press_Start_2P({
  weight: '400',
  subsets: ['latin'],
  variable: '--font-pixel',
  display: 'swap',
});

const vt323 = VT323({
  weight: '400',
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

const spaceGrotesk = Space_Grotesk({
  weight: ['400', '600', '700'],
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'POSCHAIR — Retro ROM Neo-Brutalist AI Posture Engine',
  description: 'Extreme Neo-Brutalist 16-Bit Bento Posture AI with Gemini 3.8 Flash & ElevenLabs voice coach.',
  keywords: ['posture', 'AI', 'health', 'webcam', 'voice coaching', 'neo-brutalist', 'retro arcade'],
  icons: {
    icon: '/logo.png',
    apple: '/logo.png',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${pressStart.variable} ${vt323.variable} ${spaceGrotesk.variable}`}>
      <body>{children}</body>
    </html>
  );
}
