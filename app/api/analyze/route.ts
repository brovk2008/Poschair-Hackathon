import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GEMINI_API_KEY!);

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();
    
    const modelsToTry = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-3.8-flash'];
    let text = '';
    let lastErr: unknown = null;

    for (const mName of modelsToTry) {
      try {
        const model = genAI.getGenerativeModel({
          model: mName,
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 80,
          },
        });
        const result = await model.generateContent(prompt);
        text = result.response.text().trim();
        if (text) break;
      } catch (err) {
        lastErr = err;
        console.warn(`Gemini model ${mName} unavailable, trying next...`);
      }
    }

    if (!text) {
      // Fallback deterministic biomechanical coaching cue
      text = "Roll your shoulders back, draw your chin gently inward, and align your ears over your shoulders.";
    }
    
    return NextResponse.json({ correction: text });
  } catch (error) {
    console.error('Gemini API error:', error);
    return NextResponse.json({
      correction: "Sit tall, pull your chin back, and relax your shoulders away from your ears."
    });
  }
}
