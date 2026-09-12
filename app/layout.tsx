import type { Metadata } from 'next';
import { Press_Start_2P, VT323 } from 'next/font/google';
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

export const metadata: Metadata = {
  title: 'POSCHAIR — Real-Time Posture AI',
  description: 'AI-powered posture monitoring with voice coaching. Sit better, feel better.',
  keywords: ['posture', 'AI', 'health', 'webcam', 'voice coaching'],
  openGraph: {
    title: 'POSCHAIR — Real-Time Posture AI',
    description: 'AI-powered posture monitoring with voice coaching',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${pressStart.variable} ${vt323.variable}`}>
      <body>{children}</body>
    </html>
  );
}
